#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading
import faiss

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (PendingSearchDataBatcher,
                                       EncodeResultBatchManager,
                                       SearchResultBatchManager)

from workers_util import ExecWorker, EmitWorker


SEARCH_NEXT_UDL_PREFIX = "/get_doc/"
SEARCH_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
SEARCH_NEXT_UDL_SUBGROUP_INDEX = 0



class FaissSearcher:
    def __init__(self, device: str, index_dir: str, topk: int = 5):
        self.cpu_index = None
        self.res = None
        self.gpu_index = None
        self.device = device
        self.index_dir = index_dir
        self.topk = topk
        
    def load_model(self):
        self.cpu_index = faiss.read_index(self.index_dir)
        self.res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.gpu_index.nprobe = 10
        print("Faiss index loaded")

    def searcher_exec(self, embeddings: np.ndarray) -> np.ndarray:
        if self.gpu_index is None:
            self.load_model()
        _, I = self.gpu_index.search(embeddings, self.topk)
        return I



class SearchWorker(ExecWorker):
    '''
    This is a batch executor for faiss searcher execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.searcher = FaissSearcher(self.parent.device, self.parent.index_dir, self.parent.topk)

    def create_pending_manager(self):
        return PendingSearchDataBatcher(self.max_exe_batch_size, self.parent.emb_dim)

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
                self.parent.tl.log(30030, qid, 0, batch.num_pending)
            I = self.searcher.searcher_exec(batch.embeddings[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(30031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  I, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()



class SearchEmitWorker(EmitWorker):
    '''
    This is a batcher for SearcherUDL to emit to Doc retrieve UDL
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 30100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = SEARCH_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = SEARCH_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_shards = self.parent.next_udl_shards
        self.next_udl_prefix = SEARCH_NEXT_UDL_PREFIX
        
        
        
    def create_batch_manager(self):
        # Return an instance of the batch manager that this child class needs.
        return SearchResultBatchManager()


    def add_to_buffer(self, batch, I, num_queries):
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
                                                       I[i,:])
            self.cv.notify()
        


class SearchUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.index_dir = self.conf["index_dir"]
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.next_udl_shards = self.conf.get("next_udl_shards", [0,1])
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        self.emb_dim = int(self.conf.get("emb_dim", 384))
        self.topk = int(self.conf.get("topk", 5))
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = SearchWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = SearchEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        emb_batcher = EncodeResultBatchManager()
        emb_batcher.deserialize(data)
        
        for qid in emb_batcher.question_ids:
            self.tl.log(30000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(emb_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Searcher UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
