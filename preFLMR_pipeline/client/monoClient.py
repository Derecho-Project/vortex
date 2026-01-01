#!/usr/bin/env python3

import os, time
import struct
import torch
import json
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
from serialize_utils import MonoDataBatcher

MONO_SHARD_ID = 0


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


def prepare_text_sequence(sample):
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
    # prepare inputs
    # tokenize inputs
    # pass to UDL
    tl              = TimestampLogger()
    capi            = ServiceClientAPI()
    prefix          = "/Mono/"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    batch_size      = 1
    num_batches     = 100
    
    checkpoint_path = 'LinWeizheDragon/PreFLMR_ViT-L'
    image_processor_name = 'openai/clip-vit-large-patch14'
    ds_dir = "/mydata/EVQA_datasets/EVQA_data"
    image_root_dir = "/mydata/EVQA_datasets"
    use_split = "test"
    # model configs, tokenziers
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
    
    ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + '/train-00000-of-00001.parquet',
                                            'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                            })[use_split].select([i for i in range(599)])
    # preprocess datasets so that we have 
    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_text_sequence)
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=16,
    )
    
    for i in range(0, len(ds), batch_size):
        # idx = torch.randint(0, 500, (1,)).item()
        batch = ds[i : i + batch_size]
        if (i // batch_size) >= num_batches:    
            # print(f"Batch no. {i // batch_size} reached!  Now break")
            break
        
        batcher = MonoDataBatcher()
        
        # print(f"Got qid list : {batch['question_id']}")
        for qid in batch["question_id"]:
            uds_idx =  int(qid.find("_"))
            question_id = qid[uds_idx+1:]
            batcher.question_ids.append(question_id)
        batcher.questions = batch["question"]
        batcher.text_sequence = batch["text_sequence"] 
        # print(f"Check text sequence: {batch['text_sequence']}")
        batcher.pixel_values = torch.Tensor(batch["pixel_values"]).numpy()
        # print(f"before sending, check PV size to be {torch.Tensor(batch['pixel_values']).numpy().shape}")    
        
        serialized = batcher.serialize()
        tl.log(10000, i, 0, 0)
        res = capi.put(prefix + f"_{i}", serialized.tobytes(), subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=MONO_SHARD_ID, message_id=i, trigger=True)
        
        time.sleep(5)
        
    tl.flush("mono_client_timestamp.dat")