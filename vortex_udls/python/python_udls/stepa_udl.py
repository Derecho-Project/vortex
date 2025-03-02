#!/usr/bin/env python3
import numpy as np
import json
import re
import struct
import warnings
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

import os
import torch
from torch import Tensor, nn
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel
from serialize_utils import TextDataBatcher, StepAMessageDataBatcher

STEPA_NEXT_UDL_SHARD_INDEX = 2

def deserialize_string_list(serialized_data):
    """Deserialize a custom binary format into a list of strings."""
    num_elements = struct.unpack("I", serialized_data[:4])[0]  # Read the number of elements
    offset_size = num_elements * 4  # Each offset is 4 bytes

    if num_elements == 0:
        return []

    offsets = struct.unpack(f"{num_elements}I", serialized_data[4:4 + offset_size])  # Read offsets
    string_section = serialized_data[4 + offset_size:]  # Extract string section

    # Extract strings using offsets
    string_list = []
    for i in range(num_elements):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < num_elements else len(string_section)
        string_list.append(string_section[start:end].decode('utf-8'))

    return string_list


class StepAUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepAUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        self.checkpoint_path            = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path         = '/mydata/EVQA_datasets/models/models_step_A_query_text_encoder.pt'
        self.local_projection_path      = '/mydata/EVQA_datasets/models/models_step_A_query_text_linear.pt'
        self.flmr_config                = None
        self.query_tokenizer            = None
        self.context_tokenizer          = None   
        self.query_text_encoder         = None 
        self.query_text_encoder_linear  = None
        self.device                     = 'cpu'
        self.skiplist = []
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False
        
        self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)
        
        try:
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))
        except:
            print(f'Failed to load models checkpoint!!! \n Please check {self.local_encoder_path} or {self.local_projection_path}')

    def load_model_gpu(self):
        self.device = 'cuda'
        self.query_text_encoder_linear.to(self.device)
        self.query_text_encoder.to(self.device)

    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        key = kwargs['key']
        blob = kwargs["blob"]
        
        new_batcher = TextDataBatcher()
        new_batcher.deserialize(blob)
        encoded_inputs = new_batcher.get_data()
        # blob_bytes = blob.tobytes()
        # res_json_str = blob_bytes.decode('utf-8')
        # encoded_inputs = json.loads(res_json_str)
        input_ids_np = np.copy(encoded_inputs['input_ids'])
        attn_msk_np = np.copy(encoded_inputs['attention_mask'])
        key_id = key[int(key.find('_'))+1:]
        batch_id = int(key_id)
        self.tl.log(20041, batch_id, 0, 0)
        # print('===========Step A start loading model==========')
        if self.query_text_encoder_linear == None:
            self.load_model_cpu()
            self.load_model_gpu()
        # encoded_inputs      = self.query_tokenizer(string_list)
        input_ids           = torch.LongTensor(input_ids_np).to(self.device)
        attention_mask      = torch.Tensor(attn_msk_np).to(self.device)
        # print(f"STEP A Got input ids of shape: {input_ids.shape} | attn_mask of shape: {attention_mask.shape}")
        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask,)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)
        # print('==========Step A finished forward pass==========')
        # print(f'text embedding of shape: \t {text_embeddings.shape}')
        # print(f'input ids of shape: \t\t {text_embeddings.shape}')
        # print(f'hidden sates of shape:\t{text_encoder_hidden_states.shape}')
        # result = {}
        # result['queries'] = encoded_inputs["text_sequence"]
        # result['question_id'] = encoded_inputs["question_ids"]
        # result['input_ids'] = input_ids.tolist()
        # result['text_embeddings'] = text_embeddings.tolist()
        # result['text_encoder_hidden_states'] = text_encoder_hidden_states.tolist()
        # res_json_str = json.dumps(result)
        # res_json_byte = res_json_str.encode('utf-8')
        
        stepa_serializer = StepAMessageDataBatcher()
        stepa_serializer.queries = encoded_inputs["text_sequence"]
        stepa_serializer.question_ids = encoded_inputs["question_ids"]
        stepa_serializer.input_ids = input_ids.cpu().detach().numpy()
        stepa_serializer.text_embeds = text_embeddings.cpu().detach().numpy()
        stepa_serializer.text_encoder_hidden_states = text_encoder_hidden_states.cpu().detach().numpy()
        
        stepa_serialized_np = stepa_serializer.serialize()
        subgroup_type = "VolatileCascadeStoreWithStringKey"
        subgroup_index = 0
        prefix = "/stepD/stepA_"
        
        # indices = [i for i, char in enumerate(key) if char == "/"]
        # key_id = key[int(indices[-1]):]
        
        new_key =  prefix + key_id
        res = self.capi.put(new_key, stepa_serialized_np.tobytes(), subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=STEPA_NEXT_UDL_SHARD_INDEX, message_id=1)
        self.tl.log(20050, batch_id, 0, 0)
        if batch_id==49:
            self.tl.flush(f"node{self.my_id}_STEPA_udls_timestamp.dat")
            print("STEPA TL flushed!!!")
        
        
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass