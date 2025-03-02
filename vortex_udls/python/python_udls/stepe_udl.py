#!/usr/bin/env python3
import os
import numpy as np
import json
import re
import struct
import warnings
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

import torch
from torch import Tensor, nn
from flmr import (
    FLMRConfig, 
    FLMRQueryEncoderTokenizer, 
    FLMRContextEncoderTokenizer, 
    FLMRModelForRetrieval, 
    FLMRTextModel,
    search_custom_collection, create_searcher
)
import functools
import tqdm
import tqdm
from serialize_utils import StepDMessageBatcher

class IntermediateResult:
    def __init__(self):
        self._queries           = None
        self._query_embeddings  = None
    
    def has_all(self):
        has_all = self._queries != None and self._query_embeddings != None
        return has_all


class StepEUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepEUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.searcher = None
        self.index_root_path        = '/mydata/EVQA_datasets/index/'
        self.index_experiment_name  = 'EVQA_train_split/'
        self.index_name             = 'EVQA_PreFLMR_ViT-L'
        self.collected_intermediate_results = {}
        # self.queries2save = {}
        # self.qembeds2save = torch.Tensor([], dtype = torch.float32)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        
    def load_searcher_gpu(self):
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
    
    def process_search(self, queries, query_embeddings, bsize):
        if self.searcher == None:
            self.load_searcher_gpu()
            
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=torch.Tensor(query_embeddings),
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=bsize,
        )
        
        
        return ranking.todict()
    
    
    def ocdpo_handler(self,**kwargs):
        key                 = kwargs["key"]
        blob                = kwargs["blob"]
        
        new_batcher = StepDMessageBatcher()
        new_batcher.deserialize(blob)
        cluster_result = new_batcher.get_data()
        
        # bytes_obj           = blob.tobytes()
        # json_str_decoded    = bytes_obj.decode('utf-8')
        # cluster_result      = json.loads(json_str_decoded)
        queries_texts       = cluster_result['queries']
        query_embeddings    = cluster_result['query_embeddings']
        question_ids        = cluster_result['question_ids']
        bsize               = 32
        
        
        # queries = {question_id[i]: queries_texts[i] for i in range(len(queries_texts))}
        assert len(queries_texts) == len(question_ids)
        queries = dict(zip(question_ids, queries_texts))
        
        step_D_idx = key.find('stepD')
        uds_idx = key.find("_")
        batch_id = int(key[uds_idx+1:])
        
        
        self.tl.log(40000, batch_id, 0, 0)
        ranking = self.process_search(queries, query_embeddings, bsize)
        
        # if not self.collected_intermediate_results.get(batch_id):
        #     self.collected_intermediate_results[batch_id] = IntermediateResult()
            
        # if step_D_idx != -1:
        #     self.collected_intermediate_results[batch_id]._queries = queries
        #     self.collected_intermediate_results[batch_id]._query_embeddings = query_embeddings
            
        # if not self.collected_intermediate_results[batch_id].has_all():
        #     return
            
        # ranking = self.process_search(self.collected_intermediate_results[batch_id]._queries, 
        #                               self.collected_intermediate_results[batch_id]._query_embeddings,
        #                               bsize)

        # print('==========Finished Searching==========')
        # print(f"Got queries: {self.collected_intermediate_results[batch_id]._queries}")
        # print(f'Got a ranking dictionary for batch {batch_id}: {ranking}')
        self.tl.log(40100, batch_id, 0, 0)
        
        # erase the batch id dict{} 
        # del self.collected_intermediate_results[batch_id]
        if batch_id == 49:
            self.tl.flush(f"node{self.my_id}_STEPE_udls_timestamp.dat")
            print("Time Log Flushed!!!")
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass