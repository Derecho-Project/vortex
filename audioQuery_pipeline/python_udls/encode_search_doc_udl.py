#!/usr/bin/env python3
import json
import numpy as np
from typing import Any
import threading
import faiss
from search_udl import FaissSearcher
from doc_udl import DocumentLoader
from encode_udl import TextEncoder

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (PendingEncodeDataBatcher,
                                       QueryBatcherManager,
                                       DocResultBatchManager)

from workers_util import ExecWorker, EmitWorker


SEARCH_DOC_NEXT_UDL_PREFIXES = ["/text_check/" , "/aggregate/doc_"]
SEARCH_DOC_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
SEARCH_DOC_NEXT_UDL_SUBGROUP_INDEX = 0



class EncodeSearcher:
    def __init__(self, device: str, encoder_name:str ,index_dir: str, topk: int = 5, doc_dir: str = None):
        self.encoder = TextEncoder(device, encoder_name)
        self.searcher = FaissSearcher(device, index_dir, topk)
        self.doc_loader = DocumentLoader(doc_dir)
        self.loaded_model = False        

    def load_model(self):
        self.encoder.load_model()
        self.searcher.load_model()
        self.doc_loader.load_docs()
        self.loaded_model = True

    def search_docs(self, query_list: list[str]) -> list[list[str]]:
        if not self.loaded_model:
            self.load_model()
        # Encode the queries
        embeddings = self.encoder.encoder_exec(query_list)
        # Search the embeddings
        I = self.searcher.searcher_exec(embeddings)
        # convert I to list of list of doc_ids
        doc_list_ids = []
        for i in range(I.shape[0]):
            doc_list_ids.append(I[i,:].tolist())
        # get doc_list from doc_loader
        doc_lists = self.doc_loader.get_doc_list(doc_list_ids)
        return doc_lists


class EncodeSearchDocWorker(ExecWorker):
    '''
    This is a batch executor for faiss searcher execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.encode_searcher = EncodeSearcher(self.parent.device, 
                                              self.parent.encoder_name, 
                                              self.parent.index_dir, 
                                              self.parent.topk, 
                                              self.parent.doc_dir)
        

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
            doc_list = self.encode_searcher.search_docs(batch.queries[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(20031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  doc_list, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()



class EncodeSearchDocEmitWorker(EmitWorker):
    '''
    This is a batcher for SearcherUDL to emit to Doc retrieve UDL
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 20100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = SEARCH_DOC_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = SEARCH_DOC_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_prefixes = SEARCH_DOC_NEXT_UDL_PREFIXES
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
            #  Use the grouping of next_udl_agg_shards to load balance the emit
            for i in range(num_queries):
                shard_pos = question_ids[i] % len(self.next_udl_shards)
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
                #     print(f"sent {new_key} to next UDL")
                # batch_manager.print_info()


class EncodeSearchDocUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.encoder_name = self.conf["encoder_name"]
        self.index_dir = self.conf["index_dir"]
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.next_udl_tcheck_shards = self.conf.get("next_udl_tcheck_shards", [0,1])
        self.next_udl_agg_shards = self.conf.get("next_udl_agg_shards", [0,1])
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        self.emb_dim = int(self.conf.get("emb_dim", 384))
        self.topk = int(self.conf.get("topk", 5))
        self.doc_dir = self.conf["doc_dir"]
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = EncodeSearchDocWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = EncodeSearchDocEmitWorker(self, 2)
            self.emit_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        query_batcher = QueryBatcherManager()
        query_batcher.deserialize(data)
        # query_batcher.print_info()        
    
        for qid in query_batcher.question_ids:
            self.tl.log(20000, qid, 0, 0)
            
        self.model_worker.push_to_pending_batches(query_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"EncodeSearchDoc UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
