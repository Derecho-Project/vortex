import torch, os
from torch import nn
from flmr import FLMRConfig
from transformers import BertConfig
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval

class StepC:
    def __init__(self):
        self.checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
        self.local_model_path = "/mydata/EVQA_datasets/models/models_step_C_transformer_mapping_input_linear.pt"
        self.flmr_config = None
        
        
        if not os.path.exists(self.local_model_path):
            print(f'local directory not found, initing from full model...')
            self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="query_tokenizer")
            self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                self.checkpoint_path, 
                text_config=self.flmr_config.text_config, 
                subfolder="context_tokenizer"
            )
            full_model = FLMRModelForRetrieval.from_pretrained(
                self.checkpoint_path,
                query_tokenizer=self.query_tokenizer,
                context_tokenizer=self.context_tokenizer,
            )
            self.transformer_mapping_input_linear = full_model.transformer_mapping_input_linear
            # torch.save(self.transformer_mapping_input_linear.state_dict(), self.local_model_path)

            del full_model
            
    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        
        transformer_mapping_config_base = self.flmr_config.transformer_mapping_config_base
        transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)
        transformer_mapping_config.num_hidden_layers = self.flmr_config.transformer_mapping_num_hidden_layers
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        print(f'found local model for step C, now loading...')
        self.transformer_mapping_input_linear = nn.Linear(
            self.flmr_config.vision_config.hidden_size, transformer_mapping_config.hidden_size
        )
        self.transformer_mapping_input_linear.load_state_dict(torch.load(self.local_model_path, weights_only=True))
        
    def load_model_gpu(self):
        self.transformer_mapping_input_linear.cuda()
        
    def stepC_output(self, vision_second_last_layer_hidden_states):
        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )
        
        return transformer_mapping_input_features
if __name__ == "__main__": # Bsize, vision_hidden_size[-2], vision_hidden_size[-1]
    stepc = StepC()
    stepc.load_model_cuda()
    bsize = 16
    dummy_hidden_states = torch.randn(bsize, 256, 1024).cuda()
    output = stepc.stepC_output(dummy_hidden_states)
    output.cpu()
    print(f"transformer mapping input feature shape is: {output.shape}")