#!/usr/bin/env python3
import numpy as np
import json
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
import torch
from torch import Tensor, nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer
from serialize_utils import StepDMessageBatcher, StepAMessageDataBatcher, VisionDataBatcher

STEPD_NEXT_UDL_SHARD_INDEX = 0

class IntermediateResult:
    def __init__(self):
        self._question_id       = None
        self._queries           = None
        self._input_ids         = None
        self._text_embeddings   = None
        self._text_encoder_hidden_states = None
        self._vision_embeddings = None                   
        self._transformer_mapping_input_feature = None
        
    def collected_all(self):
        has_all = self._queries != None and \
            self._input_ids != None and \
            self._text_embeddings != None and \
            self._text_encoder_hidden_states != None and \
            self._vision_embeddings != None and\
            self._transformer_mapping_input_feature != None and\
            self._question_id != None
            
        return has_all


class StepDUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepDUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        # modeling configs
        self.flmr_config = None
        self.skiplist = []
        self.query_tokenizer = None
       
        self.transformer_mapping_cross_attention_length = 32
        self.vision_encoder_embedding_size = 1024
        self.late_interaction_embedding_size = 128
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.transformer_mapping_config_base = 'bert-base-uncased'
        self.local_tf_mapping_path = '/mydata/EVQA_datasets/models/models_step_D_transformer_mapping.pt'
        self.local_tf_mapping_output_path = '/mydata/EVQA_datasets/models/models_step_D_transformer_mapping_output.pt'
        self.transformer_mapping_network = None
        self.transformer_mapping_output_linear = None
        
        self.mask_instruction = False
        
        # Kep track of collected intermediate results: {query_id0: IntermediateResult, query_id2:{} ...}
        self.collected_intermediate_results = {}
        
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        transformer_mapping_config = BertConfig.from_pretrained(self.transformer_mapping_config_base)
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        transformer_mapping_config.num_hidden_layers = 1

        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_network.load_state_dict(torch.load(self.local_tf_mapping_path, weights_only=True))
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )
        self.transformer_mapping_output_linear.load_state_dict(torch.load(self.local_tf_mapping_output_path, weights_only=True))
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
        
    def mask(self, input_ids, skiplist):
        return [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    
    
    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)

        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
        mask = [
            [
                (x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
        
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask
    
    
    def load_model_gpu(self):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()
        
    
        
    def proces_queries(self,
                       input_ids,
                       text_embeddings,
                       text_encoder_hidden_states,
                       vision_embeddings,
                       transformer_mapping_input_features,
                       ):
        if self.transformer_mapping_network == None:
            # print('==========start loading model cpu==========')
            self.load_model_cpu()
            
            # print('==========start loading model gpu==========')
            self.load_model_gpu()
            
        
        mask = torch.tensor(self.query_mask(input_ids, skiplist=self.skiplist)).unsqueeze(2).float().cuda()
        text_embeddings = text_embeddings.to(mask.device) * mask
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.transformer_mapping_cross_attention_length]
            encoder_mask = encoder_mask[:, :self.transformer_mapping_cross_attention_length]
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
        # Pass through the transformer mapping
        
        
        # ENCODER hidden states: Encoder_bsize, Encoder_seqLen, _
        # ENCODER attention mask: ones_like(encoder_hidden_states)

        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features.to(mask.device),
            encoder_hidden_states=text_encoder_hidden_states.to(mask.device),
            encoder_attention_mask=encoder_extended_attention_mask.to(mask.device),
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        # Convert the dimension to FLMR dim
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )
        # Merge with the vision embeddings
        
        vision_embeddings = torch.cat([vision_embeddings.to(mask.device), transformer_mapping_output_features], dim=1)
        
        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2).detach().cpu()
        return query_embeddings
        
        
    def ocdpo_handler(self, **kwargs):
        # preparing for input : 
            # (input_ids                              # step A      B * seq_len
            # text_embeddings                         # step A      B * 32 * 128
            # text_encoder_hidden_states              # step A      B * 32 * 768
            # vision_embeddings                       # step B      B * 32 * 128
            # transformer_mapping_input_features)     # step B (used to be C, now merged) B * 256 * 768

        
        # now we have the input formalized, proceed with the model serving
        key = kwargs["key"]
        blob = kwargs["blob"]
        
        step_A_idx = key.find("stepA") 
        step_B_idx = key.find("stepB")
        
        # print(f'Step D UDL got key: {key}')
        
        uds_idx = key.find("_")
        batch_id = int(key[uds_idx+1:])

        if not self.collected_intermediate_results.get(batch_id):
            self.collected_intermediate_results[batch_id] = IntermediateResult()
        if step_A_idx != -1:
            self.tl.log(30000, batch_id, 1, 0)

            stepa_serializer = StepAMessageDataBatcher()
            stepa_serializer.deserialize(blob)
            blob_data = stepa_serializer.get_data()
            input_ids_np = np.copy(blob_data["input_ids"])
            self.collected_intermediate_results[batch_id]._question_id = blob_data['question_ids']
            self.collected_intermediate_results[batch_id]._queries = blob_data['queries']
            self.collected_intermediate_results[batch_id]._input_ids = torch.Tensor(input_ids_np)
            self.collected_intermediate_results[batch_id]._text_embeddings = torch.Tensor(blob_data['text_embeds'])
            self.collected_intermediate_results[batch_id]._text_encoder_hidden_states = torch.Tensor(blob_data['text_encoder_hidden_states'])
            
        elif step_B_idx != -1:
            self.tl.log(30000, batch_id, 2, 0)
            stepb_batcher = VisionDataBatcher()
            stepb_batcher.deserialize(blob)
            blob_data = stepb_batcher.get_data()
            self.collected_intermediate_results[batch_id]._vision_embeddings = torch.Tensor(blob_data["vision_embedding"])
            self.collected_intermediate_results[batch_id]._transformer_mapping_input_feature = torch.Tensor(blob_data["vision_hidden_states"])
            
        if not self.collected_intermediate_results[batch_id].collected_all():
            return
        
        self.tl.log(30010, batch_id, 0, 0)
        cur_batch = self.collected_intermediate_results[batch_id]
        # call step E functions using  self.collected_intermediate_results[batch_id]
        batch_query_embeddings = self.proces_queries(   
                                    cur_batch._input_ids,
                                    cur_batch._text_embeddings,
                                    cur_batch._text_encoder_hidden_states,
                                    cur_batch._vision_embeddings,
                                    cur_batch._transformer_mapping_input_feature,
                                        )
        
        # print(f"Found batch query embeddings of shape: {batch_query_embeddings.shape}")

        self.tl.log(30011, batch_id, 0, 0)
        # self.collected_intermediate_results.erase(batch_id)
        
        # result = {}
        # result['queries'] = self.collected_intermediate_results[batch_id]._queries
        # result['query_embeddings'] = batch_query_embeddings.tolist()
        # result['question_id'] = self.collected_intermediate_results[batch_id]._question_id
        # res_json_str = json.dumps(result)
        # res_json_byte = res_json_str.encode('utf-8')
        
        stepd_batcher = StepDMessageBatcher()
        stepd_batcher.queries = self.collected_intermediate_results[batch_id]._queries
        stepd_batcher.query_embeddings = batch_query_embeddings.cpu().detach().numpy()
        stepd_batcher.question_ids = self.collected_intermediate_results[batch_id]._question_id
        stepd_batcher_np = stepd_batcher.serialize()
        
        subgroup_type = "VolatileCascadeStoreWithStringKey"
        subgroup_index = 0
        
        res = self.capi.put(f"/stepE/stepD_{batch_id}", stepd_batcher_np.tobytes(), subgroup_type=subgroup_type,
                subgroup_index=subgroup_index,shard_index=STEPD_NEXT_UDL_SHARD_INDEX, message_id=1, as_trigger=True, blocking=False)

        self.tl.log(30020, batch_id, 0, 0)
        
        if batch_id == 49:
            self.tl.flush(f"node{self.my_id}_STEPD_udls_timestamp.dat")
            print("StepD TL flushed!!!")
        # garbage cleaning via emit and del
        del self.collected_intermediate_results[batch_id]
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass