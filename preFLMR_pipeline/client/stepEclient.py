#!/usr/bin/env python3

import numpy as np
import os
import torch
from derecho.cascade.external_client import ServiceClientAPI
import struct
from PIL import Image
import sys
import json


if __name__ == "__main__":
    capi            = ServiceClientAPI()
    key             = "/stepE/1"
    subgroup_type   = "VolatileCascadeStoreWithStringKey"
    subgroup_index  = 0
    shard_index     = 0
    
    
    num_queries = 2
    dummy_q_embeds = torch.randn(num_queries, 320, 128).tolist()
    query_instructions = [f"instruction {i}" for i in range(num_queries)]
    query_texts = [f"{query_instructions[i]} : query {i}" for i in range(num_queries)]
    queries = {i: query_texts[i] for i in range(num_queries)}

    dummy_dict = {
        'queries': queries,
        'Qembeddings': dummy_q_embeds,
    }
    json_string = json.dumps(dummy_dict)
    byte_data = json_string.encode('utf-8')
    
    res = capi.put(key, byte_data, subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)