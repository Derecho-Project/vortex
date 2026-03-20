#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading
import torch

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (DocResultBatchManager, 
                                       PendingCheckDataBatcher,
                                       TextCheckResultBatchManager)
from transformers import BartTokenizer, BartForSequenceClassification
from workers_util import ExecWorker, EmitWorker


CHECK_NEXT_UDL_PREFIX = "/aggregate/check_"
CHECK_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
CHECK_NEXT_UDL_SUBGROUP_INDEX = 0


class TextChecker:
    def __init__(self, device: str, model_name: str):
        self.model = None
        self.tokenizer = None
        self.device = device
        self.model_name = model_name
        self.hypothesis = "harmful."
        
    def load_model(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        print("Text Check model loaded")

    def model_exec(self, batch_premise: list[str]) -> np.ndarray:
        '''
        batch_premise: list of text strings
        
        return: list of integer. probability in % that the content is harmful
        '''
        if self.model is None:
            self.load_model()
        inputs = self.tokenizer(batch_premise,
                       [self.hypothesis] * len(batch_premise),
                       return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)
        logits = result.logits
        entail_contradiction_logits = logits[:, [0, 2]]  # entailment = index 2
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100  # entailment probability
        true_probs = [int(x) for x in true_probs]
        return true_probs
    
    def docs_check(self, doc_list: list[list[str]]) -> list[list[int]]:
        flattened_doc_list = [item for sublist in doc_list for item in sublist]
        types = self.model_exec(flattened_doc_list)
        # Reshape the types to match the original doc_list structure
        reshaped_types = []
        start = 0
        for sublist in doc_list:
            end = start + len(sublist)
            reshaped_types.append(types[start:end])
            start = end
        return reshaped_types



class TextCheckWorker(ExecWorker):
    '''
    This is a batch executor for text checker execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.checker_model = TextChecker(self.parent.device, self.parent.model_name)

    def create_pending_manager(self):
        return PendingCheckDataBatcher(self.max_exe_batch_size)

    def main_loop(self):
        batch = None
        while self.running:
            if not batch is None:
                batch.reset()
            with self.cv:
                self.current_batch = -1
                if self.pending_batches[self.next_to_process].num_pending == 0:
                    self.cv.wait(timeout=self.batch_time_us/1000000)   
                if self.pending_batches[self.next_to_process].num_pending != 0:
                    self.current_batch = self.next_to_process
                    self.next_to_process = (self.next_to_process + 1) % len(self.pending_batches)
                    batch = self.pending_batches[self.current_batch]
                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    self.new_space_available = True
                    self.cv.notify()
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            # Execute the batch
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(50030, qid, 0, batch.num_pending)
            doc_types = self.checker_model.docs_check(batch.doc_list[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(50031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  doc_types, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()



class TextCheckEmitWorker(EmitWorker):
    '''
    This is a batcher for checkerUDL to emit to aggregate UDL
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 50100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = CHECK_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = CHECK_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_shards = self.parent.next_udl_shards
        self.next_udl_prefix = CHECK_NEXT_UDL_PREFIX
        
        
        
    def create_batch_manager(self):
        # Return an instance of the batch manager that this child class needs.
        return TextCheckResultBatchManager()


    def add_to_buffer(self, batch, doc_types, num_queries):
        '''
        pass by object reference to avoid deep-copy
        '''
        question_ids = batch.question_ids[:num_queries]
        
        with self.cv:
            # use question_id to determine which shard to send to
            for i in range(num_queries):
                shard_pos = question_ids[i] % len(self.parent.next_udl_shards)
                self.send_buffer[shard_pos].add_result(question_ids[i],  
                                                       doc_types[i])
            self.cv.notify()
        


class TextCheckUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.model_name = self.conf["model_name"]
        # Note: batch size here is topk(5) * max_exe_batch_size, 
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.next_udl_shards = self.conf.get("next_udl_shards", [0,1])
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = TextCheckWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = TextCheckEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        doc_result_batcher = DocResultBatchManager()
        doc_result_batcher.deserialize(data)
        
        for qid in doc_result_batcher.question_ids:
            self.tl.log(50000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(doc_result_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Checker UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
