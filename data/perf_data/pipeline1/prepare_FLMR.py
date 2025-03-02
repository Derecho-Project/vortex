from flmr import (
    FLMRConfig, 
    FLMRVisionModel, 
    FLMRContextEncoderTokenizer, 
    FLMRModelForRetrieval, 
    FLMRTextModel, 
    FLMRQueryEncoderTokenizer, 
    FLMRContextEncoderTokenizer, FLMRModelForRetrieval,
    ) 
from transformers import AutoImageProcessor


if __name__ == "__main__":
    
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                    checkpoint_path, 
                    text_config=flmr_config.text_config, 
                    subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
                    checkpoint_path, 
                    text_config=flmr_config.text_config, 
                    subfolder="context_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    flmr_model = FLMRModelForRetrieval.from_pretrained(
        checkpoint_path,
        query_tokenizer = query_tokenizer,
        context_tokenizer = context_tokenizer,
        )
    