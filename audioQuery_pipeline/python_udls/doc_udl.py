#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading
import faiss
import pickle

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (PendingLoaderDataBatcher,
                                       SearchResultBatchManager,
                                       DocResultBatchManager)

from workers_util import ExecWorker, EmitWorker


DOC_NEXT_UDL_PREFIXES = ["/text_check/" , "/lang_det/", "/aggregate/doc_"]
DOC_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
DOC_NEXT_UDL_SUBGROUP_INDEX = 0


class DocumentLoader:
    def __init__(self, doc_dir):
        self.doc_dir = doc_dir
        self.doc_list = None
        
    def load_docs(self):
        with open(self.doc_dir , 'rb') as file:
            self.doc_list = pickle.load(file)
        print("Document list loaded")


    def get_doc_list(self, doc_ids_list) -> list[list[str]]:
        '''
        doc_ids_list: list of list of doc_ids
        '''
        if self.doc_list is None:
            self.load_docs()
        doc_lists = []
        for doc_ids in doc_ids_list:
            cur_docs = []
            for doc_id in doc_ids:
                cur_docs.append(self.doc_list[doc_id])
            doc_lists.append(cur_docs)
        return doc_lists



class DocWorker(ExecWorker):
    '''
    This is a batch executor for faiss searcher execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.doc_loader = DocumentLoader(self.parent.doc_dir)

    def create_pending_manager(self):
        return PendingLoaderDataBatcher(self.max_exe_batch_size)

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
                self.parent.tl.log(40030, qid, 0, batch.num_pending)
            doc_lists = self.doc_loader.get_doc_list(batch.doc_ids[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(40031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  doc_lists, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()



class DocEmitWorker(EmitWorker):
    '''
    This is a batcher for SearcherUDL to emit to text check and language detection UDLs
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 40100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = DOC_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = DOC_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_shards = self.parent.next_udl_shards
        self.next_udl_prefixes = DOC_NEXT_UDL_PREFIXES
        #  Use the grouping of next_udl_agg_shards to setup the emit batch
        self.next_udl_shards = self.parent.next_udl_agg_shards
        
        
        
    def create_batch_manager(self):
        # Return an instance of the batch manager that this child class needs.
        return DocResultBatchManager()


    def add_to_buffer(self, batch, doc_lists, num_queries):
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
                                                       doc_lists[i])
            self.cv.notify()
    
    def process_and_emit_results(self, to_send):
        for idx, batch_manager in enumerate(to_send):
            if batch_manager.num_queries == 0:
                continue
            num_sent = 0
            cur_shard_id = self.next_udl_shards[idx]
            while num_sent < batch_manager.num_queries:
                serialize_batch_size = min(self.max_emit_batch_size, batch_manager.num_queries - num_sent)
                start_pos = num_sent
                end_pos = num_sent + serialize_batch_size
                serialized_batch = batch_manager.serialize(start_pos, end_pos)  

                for next_udl_prefix in self.next_udl_prefixes:
                    new_key = next_udl_prefix + str(self.parent.sent_msg_count) + "_" + str(cur_shard_id)
                    if next_udl_prefix == "/aggregate/doc_":
                        shard_idx = self.parent.next_udl_agg_shards[idx]
                    elif next_udl_prefix == "/text_check/":
                        shard_idx = self.parent.next_udl_tcheck_shards[idx % len(self.parent.next_udl_tcheck_shards)]
                    elif next_udl_prefix == "/lang_det/":
                        shard_idx = self.parent.next_udl_lang_shards[idx % len(self.parent.next_udl_lang_shards)]
                    
                    for qid in batch_manager.question_ids[start_pos:end_pos]:
                        self.parent.tl.log(self.emit_log_flag, qid, 0, end_pos - start_pos)
                    
                    self.parent.capi.put_nparray(new_key, serialized_batch, 
                                    subgroup_type=self.next_udl_subgroup_type,
                                    subgroup_index=self.next_udl_subgroup_index, 
                                    shard_index=shard_idx, 
                                    message_id=1, as_trigger=True, blocking=False)
                    self.parent.sent_msg_count += 1
                    num_sent += serialize_batch_size
                    self.sent_batch_counter += 1
                    


class DocUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.doc_dir = self.conf["doc_dir"]
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        self.next_udl_tcheck_shards = self.conf.get("next_udl_tcheck_shards", [0,1])
        self.next_udl_lang_shards = self.conf.get("next_udl_lang_shards", [0,1])
        self.next_udl_agg_shards = self.conf.get("next_udl_agg_shards", [0,1])
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = DocWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = DocEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        search_res_batcher = SearchResultBatchManager()
        search_res_batcher.deserialize(data)
        
        for qid in search_res_batcher.question_ids:
            self.tl.log(40000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(search_res_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Loader UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
