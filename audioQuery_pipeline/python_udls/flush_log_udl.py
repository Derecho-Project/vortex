#!/usr/bin/env python3

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger



class FlushLogUDL(UserDefinedLogic):
     '''
     FlushLogUDL is the UDL to flush the timestamp log for all UDLs in this node
     '''
     def __init__(self,conf_str):
          '''
          Constructor
          '''
          super(FlushLogUDL,self).__init__(conf_str)
          self.capi = ServiceClientAPI()
          self.my_id = self.capi.get_my_id()
          self.tl = TimestampLogger()

     def ocdpo_handler(self,**kwargs):
          '''
          The off-critical data path handler
          '''
          self.tl.flush(f"node{self.my_id}_timestamp.dat")
          print(f"node{self.my_id}")
          

     def __del__(self):
          '''
          Destructor
          '''
          print(f"FlushLogUDL destructor")
          pass