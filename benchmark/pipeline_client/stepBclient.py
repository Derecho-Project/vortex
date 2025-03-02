#!/usr/bin/env python3

import numpy as np
import os
import torch
from derecho.cascade.external_client import ServiceClientAPI
import struct
from transformers import AutoImageProcessor
from PIL import Image
import sys

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


if __name__ == "__main__":
    
    capi            = ServiceClientAPI()
    prefix          = "/stepB/"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    shard_index     = 0
    
    img_root        = './perf_data/pipeline1/data/images'
    image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-large-patch14', use_fast=True)
    processed_img_nparray = process_img_2_nparray(img_root, image_processor)

    print(f"got ndarray of size {sys.getsizeof(processed_img_nparray)}")
    
    for i in range(10):
        key = prefix + f'_{i}'
        res = capi.put(key, processed_img_nparray, subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)