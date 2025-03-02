#!/usr/bin/env python3
import numpy as np
import json
import re
import struct

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI

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


class ConsolePrinterUDL(UserDefinedLogic):
     '''
     ConsolePrinter is the simplest example showing how to use the udl
     '''
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(ConsolePrinterUDL,self).__init__(conf_str)
          self.conf = json.loads(conf_str)
          print(f"ConsolePrinter constructor received json configuration: {self.conf}")
          pass

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler
          '''
          blob = kwargs["blob"]
          string_list = deserialize_string_list(blob.tobytes())
          print(f"I recieved kwargs: {string_list}")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"ConsolePrinterUDL destructor")
          pass