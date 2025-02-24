#!/usr/bin/env python3

import numpy as np
import os, sys
import struct
import torch
import json
from collections import defaultdict
from easydict import EasyDict
from derecho.cascade.external_client import ServiceClientAPI
from derecho.cascade.external_client import TimestampLogger
from transformers import AutoImageProcessor
from PIL import Image
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from datasets import load_dataset
import time



STEPA_SHARD_INDEX = 3
STEPB_SHARD_INDEX = 1
STEPA_SUBGROUP_INDEX = 0
STEPB_SUBGROUP_INDEX = 0

def process_img_2_nparray(img_root, image_processor):
    img_paths = [os.path.join(img_root, item) for item in os.listdir(img_root)]
    list_of_images = []
    pixel_values = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        list_of_images.append(image)
   
    for img in list_of_images:
        encoded = image_processor(img, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)
    pixel_values = torch.stack(pixel_values, dim=0)

    return pixel_values.numpy().tobytes()


def serialize_string_list(string_list):
    """Serialize a list of strings into a custom binary format."""
    encoded_strings = [s.encode('utf-8') for s in string_list]
    offsets = []
    current_offset = 0

    # Compute the offsets for each string
    for s in encoded_strings:
        offsets.append(current_offset)
        current_offset += len(s)

    # Pack the number of elements
    header = struct.pack("I", len(string_list))  # 4 bytes for length
    offset_section = struct.pack(f"{len(offsets)}I", *offsets)  # 4 bytes per offset

    # Concatenate everything into a byte stream
    serialized_data = header + offset_section + b''.join(encoded_strings)
    return serialized_data


def prepare_inputs(sample):
    sample = EasyDict(sample)

    module = EasyDict(
        {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
    )

    instruction = sample.instruction.strip()
    if instruction[-1] != ":":
        instruction = instruction + ":"
    instruction = instruction.replace(":", flmr_config.mask_instruction_token)
    #random_instruction = random.choice(instructions)
    text_sequence = " ".join(
        [instruction]
        + [module.separation_tokens.start]
        + [sample.question]
        + [module.separation_tokens.end]
    )

    sample["text_sequence"] = text_sequence

    return sample
    
    
def tokenize_inputs(examples, query_tokenizer, image_processor):
        encoding = query_tokenizer(examples["text_sequence"])
        examples["input_ids"] = encoding["input_ids"]
        examples["attention_mask"] = encoding["attention_mask"]

        pixel_values = []
        for img_path in examples["img_path"]:

            if img_path is None:
                image = Image.new("RGB", (336, 336), color='black')
            else:
                image = Image.open(img_path).convert("RGB")
            
            encoded = image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)

        pixel_values = torch.stack(pixel_values, dim=0)
        examples["pixel_values"] = pixel_values
        return examples
    
def add_path_prefix_in_img_path(example, prefix):
        if example["img_path"] != None:
            example["img_path"] = os.path.join(prefix, example["img_path"])
        return example
    
    

if __name__ == "__main__":
    
    tl = TimestampLogger()
    capi = ServiceClientAPI()
    stepa_prefix = "/stepA/"
    stepb_prefix = "/stepB/"
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    
    batch_size = 1
    num_batches = 100
    
    # directories and str configs
    image_processor_name = 'openai/clip-vit-large-patch14'
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    image_root_dir = "/mydata/EVQA_datasets"
    use_split = "train"
    ds_dir = "/mydata/EVQA_datasets/EVQA_data"
    # model configs, tokenziers
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    
    
    ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + '/train-00000-of-00001.parquet',
                                            'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                            })[use_split].select(i for i in range(999))
    # preprocess datasets so that we have 
    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_inputs)
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=16,
    )
    
    # slice the ds so that we split the input are
    
    
    for i in range(0, len(ds), batch_size):
        idx = torch.randint(0, 555, (1,)).item()
        batch = ds[idx : idx + batch_size]
        # print(f"got batch {batch} with idx being {idx} and {idx+batch_size}")
        if (i // batch_size) >= num_batches:    
            # print(f"Batch no. {i // batch_size} reached!  Now break")
            break
        
        # print(f"Check for input ids: {torch.LongTensor(batch['input_ids']).shape} | \n attention_mask: {torch.Tensor(batch['attention_mask']).shape}")
            
        stepa_data2send_keys = ["question_id", "text_sequence", "input_ids", "attention_mask"]
        stepa_data2send_dict = {k: batch[k].numpy() if isinstance(batch[k], torch.Tensor) or isinstance(batch[k], torch.LongTensor) else batch[k] for k in stepa_data2send_keys if k in batch}
        stepa_key = stepa_prefix + f"_{i}"
        stepa_json_str = json.dumps(stepa_data2send_dict)
        stepa_byte_data = stepa_json_str.encode('utf-8')
        tl.log(10000 ,i ,0 ,0 )
        resA = capi.put(stepa_key, stepa_byte_data,subgroup_type=subgroup_type,
                    subgroup_index=STEPA_SUBGROUP_INDEX,shard_index=STEPA_SHARD_INDEX, message_id=1, as_trigger=True, blokcing=True)
        

        stepb_data2send_keys = ["pixel_values"]
        stepb_data2send_dict = {k: batch[k].numpy() if isinstance(batch[k], torch.Tensor) else batch[k] for k in stepb_data2send_keys if k in batch}
        stepb_key = stepb_prefix + f"_{i}"
        stepb_json_str = json.dumps(stepb_data2send_dict)
        stepb_byte_data = stepb_json_str.encode('utf-8')
        
        resB = capi.put(stepb_key, stepb_byte_data,subgroup_type=subgroup_type,
                    subgroup_index=STEPB_SUBGROUP_INDEX,shard_index=STEPB_SHARD_INDEX, message_id=1, trigger=True)
        time.sleep(2)
        
    tl.flush("client_timestamp.dat")
        # time.sleep(15)
    # for i in range(10):
    #     key = prefix + f"_{i}"
    #     res = capi.put(key, serialize_string_list(value),subgroup_type=subgroup_type,
    #                 subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)



