import json
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
import torch
from easydict import EasyDict
import numpy as np
from transformers import AutoImageProcessor
from flmr import (
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRConfig,
)
from flmr import create_searcher, search_custom_collection
from serialize_utils import MonoDataBatcher


class Monolithic_UDL(UserDefinedLogic):
    
    def __init__(self,conf_str):
        super(Monolithic_UDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.index_root_path        = '/mydata/EVQA_datasets/index/'
        self.index_name             = 'EVQA_PreFLMR_ViT-L'
        self.index_experiment_name  = 'EVQA_test_split'
        self.checkpoint_path        = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.image_processor_name   = 'openai/clip-vit-large-patch14'
        self.Ks                     = [1]
        self.use_gpu                = True
        self.nbits                  = 8
        self.query_batch_size       = 8
        self.flmr_config            = None
        self.query_tokenizer        = None
        self.context_tokenizer      = None
        self.flmr_model             = None
        self.image_processor        = None
        self.searcher               = None
        
        
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="context_tokenizer")
        self.flmr_model = FLMRModelForRetrieval.from_pretrained(
            self.checkpoint_path,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.image_processor_name)
        
        
    def load_model_gpu(self):
        self.flmr_model = self.flmr_model.to("cuda")
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=self.nbits, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )
        
    
    def ocdpo_handler(self,**kwargs):
        bsize               = 32
        key                 = kwargs["key"]
        blob                = kwargs["blob"]
        # bytes_obj           = blob.view(dtype=np.uint8)
        # json_str_decoded    = bytes_obj.decode('utf-8')
        new_batcher = MonoDataBatcher()
        new_batcher.deserialize(blob)
        data = new_batcher.get_data()
        np_pixel_vals = np.copy(data["pixel_values"])
        
        
        
        if self.flmr_model == None:
            self.load_model_cpu()
            
        if self.searcher == None:
            self.load_model_gpu()
        
        examples = {}
        examples["pixel_values"] = torch.Tensor(np_pixel_vals)
        # print(f"GOT PV shape: {examples['pixel_values'].shape}")
        examples["question_id"] = []
        for qid in data["question_ids"]:
            examples["question_id"].append(f"EVQA_{qid}")
        examples["question"] = data["questions"]
        examples["text_sequence"] = data["text_sequence"]
        
        
        encoding = self.query_tokenizer(examples["text_sequence"])
        input_ids = torch.LongTensor(encoding["input_ids"]).to("cuda")
        attention_mask = torch.LongTensor(encoding["attention_mask"]).to("cuda")
        pixel_values = torch.FloatTensor(examples["pixel_values"]).to("cuda")
        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        query_embeddings = self.flmr_model.query(**query_input).late_interaction_output
        query_embeddings = query_embeddings.detach().cpu()
        
        # print(f"Got Qembeds of size: {query_embeddings.shape}")
        
        queries = {
            question_id: question for question_id, question in zip(examples["question_id"], examples["question"])
        } 
        
        self.tl.log(40000, int(kwargs["message_id"]), 0, 0)
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries= queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=32,
        )
        
        # print(f'For batch number {int(kwargs["message_id"])} got ranking dictionary: {ranking.todict()}')
        self.tl.log(40100, int(kwargs["message_id"]), 0, 0)
        # if int(kwargs["message_id"]) % 5 ==0:
        #     print(f"Now processing batch no.{int(kwargs['message_id'])}")
        #     print(f"Got ranking dict: {ranking.todict()}")
            
        if int(kwargs["message_id"]) == 99:
            self.tl.flush(f"mono_node_{self.my_id}_timestamp.dat")
            print("TL flushed!!!")
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass