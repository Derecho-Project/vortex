#!/usr/bin/env python3
import json
import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger
import numpy as np
import torch
from torch import nn
from flmr import FLMRConfig, FLMRVisionModel
from transformers import AutoImageProcessor
from step_C_modeling_mlp import StepC
from serialize_utils import PixelValueBatcher
from serialize_utils import VisionDataBatcher


STEPB_NEXT_UDL_SHARD_INDEX = 2

class FLMRMultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron with an activation function. This can be used as the mapping network in the FLMR model.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(FLMRMultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        
        
class StepBUDL(UserDefinedLogic):
    '''
    ConsolePrinter is the simplest example showing how to use the udl
    '''
    def __init__(self,conf_str):
        '''
        Constructor
        '''
        super(StepBUDL,self).__init__(conf_str)
        self.conf = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.my_id = self.capi.get_my_id()
        self.tl = TimestampLogger()
        # print(f"ConsolePrinter constructor received json configuration: {self.conf}")
        
        self.checkpoint_path            = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_encoder_path         = '/mydata/EVQA_datasets/models/models_step_B_vision_encoder.pt'
        self.local_projection_path      = '/mydata/EVQA_datasets/models/models_step_B_vision_projection.pt'
        self.image_processor            = None
        self.flmr_config                = None

        self.query_vision_encoder       = None
        self.query_vision_projection    = None
        self.skiplist = []
        self.device = None
        self.stepc = StepC()
        
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
        self.query_vision_projection = FLMRMultiLayerPerceptron(
                (
                    self.flmr_config.vision_config.hidden_size,
                    (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length) // 2,
                    self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
                )
            )
        self.query_vision_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=False))
        self.query_vision_projection.load_state_dict(torch.load(self.local_projection_path, weights_only=False))
    
    def load_model_gpu(self):
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()
        self.device = self.query_vision_encoder.device
        
    def ocdpo_handler(self,**kwargs):
        '''
        The off-critical data path handler
        '''
        key = kwargs["key"]
        blob = kwargs["blob"]
        
        new_batcher = PixelValueBatcher()
        new_batcher.deserialize(blob)
        data = new_batcher.get_data()
        
        pv_np = np.copy(data["pixel_values"])
        key_id = key[int(key.find('_'))+1:]
        batch_id = int(key_id)
        self.tl.log(20051, batch_id, 0, 0)
        # print(f"GOT MY ID AS: {self.my_id}")
        # list_of_images = deserialize_string_list(blob.tobytes())
        # should be a 5D tensor of shape B * 1 * n_channel(3) * H * W
        # reconstructed_np_array = np.frombuffer(blob, dtype=np.float32).reshape(-1, 1, 3, 224, 224)
        # input_tensor = torch.Tensor(reconstructed_np_array)
        
        # print('==========Step B+C start loading model==========')

        if self.query_vision_projection == None:
            self.load_model_cpu()
            self.stepc.load_model_cpu()
            self.load_model_gpu()
            self.stepc.load_model_gpu()
            
        input_tensor = torch.Tensor(pv_np).to(self.device)
        # print(f"STEP B Got input tensor of shape: {input_tensor.shape}")
        batch_size = input_tensor.shape[0]
        # Forward the vision encoder
        if len(input_tensor.shape) == 5:
            # Multiple ROIs are provided
            # merge the first two dimensions
            input_tensor = input_tensor.reshape(
                -1, input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]
            )
        vision_encoder_outputs = self.query_vision_encoder(input_tensor, output_hidden_states=True)
        vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]
        vision_embeddings = self.query_vision_projection(vision_embeddings)
        vision_embeddings = vision_embeddings.view(batch_size, -1, self.flmr_config.dim)
    
        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
        
        # print('==========Step B Finished ==========')        
        transformer_mapping_input_feature = self.stepc.stepC_output(vision_second_last_layer_hidden_states)
        # print(f'the shape of hs: {transformer_mapping_input_feature.shape} | and for ve: {vision_embeddings.shape}')
        # print('==========Step C Finished ==========')
        # ve_result = {}
        # hs_result = {}
        # ve_result['vision_embeddings'] = vision_embeddings.tolist()
        # hs_result['transformer_mapping_input_feature'] = transformer_mapping_input_feature.tolist()
        # veres_json_str = json.dumps(ve_result)
        # veres_json_byte = veres_json_str.encode('utf-8')
        # hsres_json_str = json.dumps(hs_result)
        # hsres_json_byte = hsres_json_str.encode('utf-8')
        stepb_batcher = VisionDataBatcher()
        
        ve = vision_embeddings.detach().cpu().numpy()
        hs = transformer_mapping_input_feature.detach().cpu().numpy()
        stepb_batcher.vision_embedding = ve
        stepb_batcher.vision_hidden_states = hs
        stepb_batcher.question_id = data["question_ids"].tolist()
        stepb_bacher_np = stepb_batcher.serialize()
        
        prefix = "/stepD/stepB_"

        # indices = [i for i, char in enumerate(key) if char == "/"]
        # key_id = key[int(indices[-1]):]
        
        key = prefix + key_id
        subgroup_type = "VolatileCascadeStoreWithStringKey"
        subgroup_index = 0
        
        resve = self.capi.put(key,stepb_bacher_np.tobytes(),subgroup_type=subgroup_type,
                      subgroup_index=subgroup_index,shard_index=STEPB_NEXT_UDL_SHARD_INDEX,
                      message_id=1, as_trigger=True, blocking=False)
        
        # reshs = self.capi.put(hs_key,hs,subgroup_type=subgroup_type,
        #               subgroup_index=subgroup_index,shard_index=STEPB_NEXT_UDL_SHARD_INDEX,
        #               message_id=1, as_trigger=True, blocking=False)
        self.tl.log(20100, batch_id, 0, 0)
        
        if batch_id == 49:
            self.tl.flush(f"node{self.my_id}_STEPB_udls_timestamp.dat")
            print("STEPB TL flushed!!!")
        
    def __del__(self):
        '''
        Destructor
        '''
        print(f"ConsolePrinterUDL destructor")
        pass
    