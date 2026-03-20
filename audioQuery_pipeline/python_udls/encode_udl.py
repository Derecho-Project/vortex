#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (QueryBatcherManager, 
                                       PendingEncodeDataBatcher,
                                       EncodeResultBatchManager)
# from FlagEmbedding import FlagModel
from sentence_transformers import SentenceTransformer
from workers_util import ExecWorker, EmitWorker


ENCODE_NEXT_UDL_PREFIX = "/search/"
ENCODE_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
ENCODE_NEXT_UDL_SUBGROUP_INDEX = 0


class TextEncoder:
    def __init__(self, device: str, model_name: str):
        self.encoder = None
        self.device = device
        self.model_name = model_name
        
    def load_model(self):
        self.encoder = SentenceTransformer(self.model_name, devices=self.device)
        print("Text Encoder model loaded")

    def encoder_exec(self, query_list: list[str]) -> np.ndarray:
        # Generate embedding dimesion of 384
        if self.encoder is None:
            self.load_model()
        result =  self.encoder.encode(query_list)
        return result



class EncodeWorker(ExecWorker):
    '''
    This is a batch executor for text encoder execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.encoder_model = TextEncoder(self.parent.device, self.parent.model_name)

    def create_pending_manager(self):
        return PendingEncodeDataBatcher(self.max_exe_batch_size)

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
                self.parent.tl.log(20030, qid, 0, batch.num_pending)
            embeddings = self.encoder_model.encoder_exec(batch.queries[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(20031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  embeddings, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()



class EncodeEmitWorker(EmitWorker):
    '''
    This is a batcher for encoderUDL to emit to centroids search UDL
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 20100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = ENCODE_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = ENCODE_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_shards = self.parent.next_udl_shards
        self.next_udl_prefix = ENCODE_NEXT_UDL_PREFIX
        
        
        
    def create_batch_manager(self):
        # Return an instance of the batch manager that this child class needs.
        return EncodeResultBatchManager()


    def add_to_buffer(self, batch, embeddings, num_queries):
        '''
        pass by object reference to avoid deep-copy
        '''
        question_ids = batch.question_ids[:num_queries]
        queries = batch.queries[:num_queries]
        
        with self.cv:
            # use question_id to determine which shard to send to
            for i in range(num_queries):
                shard_pos = question_ids[i] % len(self.parent.next_udl_shards)
                self.send_buffer[shard_pos].add_result(question_ids[i],  
                                                       queries[i],
                                                       embeddings[i])
            self.cv.notify()
        


class EncodeUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.model_name = self.conf["model_name"]
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
            self.model_worker = EncodeWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = EncodeEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        query_batcher = QueryBatcherManager()
        query_batcher.deserialize(data)
        
        for qid in query_batcher.question_ids:
            self.tl.log(20000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(query_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Encoder UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
