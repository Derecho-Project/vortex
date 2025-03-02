#!/usr/bin/env python3

import numpy as np
import os
from derecho.cascade.external_client import ServiceClientAPI

import struct

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



    

if __name__ == "__main__":
    capi = ServiceClientAPI()
    prefix = "/stepA/"
    value = ["GOJI is my puppy."]
    subgroup_type = "VolatileCascadeStoreWithStringKey"
    subgroup_index = 0
    shard_index = 0

    for i in range(10):
        key = prefix + f"_{i}"
        res = capi.put(key, serialize_string_list(value),subgroup_type=subgroup_type,
                    subgroup_index=subgroup_index,shard_index=shard_index, message_id=1)



