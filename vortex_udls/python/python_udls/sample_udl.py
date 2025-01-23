#!/usr/bin/env python3
from derecho.cascade.udl import UserDefinedLogic
import cascade_context
import numpy as np
import json
import re
from derecho.cascade.member_client import ServiceClientAPI

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
          # cascade_context.emit( ... )
          print(f"I recieved kwargs: {kwargs}")

     def __del__(self):
          '''
          Destructor
          '''
          print(f"ConsolePrinterUDL destructor")
          pass
