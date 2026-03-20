#!/usr/bin/env python3
import json
import numpy as np
import pickle
import torch
import transformers
from typing import MutableSequence, Dict

import cascade_context
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
from derecho.cascade.udl import UserDefinedLogic
from pipeline2_serialize_utils import DocGenResult, DocGenResultBatcher, AggregateResultBatch

NEXT_UDL_PREFIXES = ["/rag/generate/check"]
NEXT_UDL_SUBGROUP_INDEXES = [1]

class DocGenerateUDL(UserDefinedLogic):
    """
    This UDL is used to retrieve documents and generate response using LLM.
    """

    def __init__(self,conf_str):
        '''
        Constructor
        '''
        # collect the LLM result per client_batch {(query_batch_key,query_count):{query_id: LLMResult, ...}, ...}
        self.llm_res = {}
        self.conf = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.doc_file_name = self.conf["doc_file_name"]
        self.doc_content_list = None
        self.pipeline = None
        self.terminators = None
        print("[DocGenerateUDL] initialized")
        # self.pending_queries = [] # list of AggregateResultBatch


    def load_llm(self,):
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
        print("[DocGenerateUDL] loaded LLM model")
        
    
    def load_doc(self):
        """
        Helper method to load the document content list to memory.
        """
        with open(self.doc_file_name, 'rb') as f:
            self.doc_content_list = pickle.load(f)
        # print(f"Loaded {len(self.doc_content_list)} documents")
          


    def retrieve_documents(self, queries: MutableSequence[Dict[str, any]]) -> MutableSequence[DocGenResult]:
        """
        batch retrieve documents based on search results
        @queries: list of queries. list and dictionary are mutable objects in python, passed by reference
        @return: list of DocGenResult objects
        """     
        result_list = []
        if not self.doc_content_list:
            self.load_doc()
        for query in queries:            
            result_list.append(DocGenResult(query['query_id'], query['client_id'], query['text'], [self.doc_content_list[doc_id] for doc_id in query['doc_ids']]))
        return result_list


    def llm_generate(self, doc_gen_results: MutableSequence[DocGenResult]):
        """
        @doc_gen_results: list of DocGenResult objects, containing query text, context list
        """    
        if not self.pipeline:
            self.load_llm()
            
        for query in doc_gen_results:
            messages = [
                {"role": "system", "content": "Answer the user query based on this list of documents:"+" ".join(query.context)},
                {"role": "user", "content": query.text},
            ]
            
            llm_result = self.pipeline(
                messages,
                max_new_tokens=256,
                #eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            query.response = llm_result[0]["generated_text"][-1]['content']
            
     
    def ocdpo_handler(self,**kwargs):
        key = kwargs["key"]
        blob = kwargs["blob"]
        binary_data = np.frombuffer(blob, dtype=np.uint8)
        agg_results = AggregateResultBatch(binary_data)
        batched_queries = agg_results.get_queries(decode_texts=True)
        
        doc_gen_results = self.retrieve_documents(batched_queries)
        self.llm_generate(doc_gen_results)

        # serialize the result
        result_batcher = DocGenResultBatcher()
        for result in doc_gen_results:
            result_batcher.add_doc_gen_result(result)
            
        result_batcher.serialize_response()
        
        # emit to the next UDL
        new_key = NEXT_UDL_PREFIXES[0] + f"/{key}"
        cascade_context.emit(new_key, result_batcher._bytes)
        # print(f" [DocGenerateUDL] emitted {key} results to {new_key}")

          

    def __del__(self):
        pass
    