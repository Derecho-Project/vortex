# Python UDL documentation

## Organization

Python_udl_manager is a DLL library that manages all python UDLs. Python udls are defined in python files. To allow python_udl_manager to load the functions defined in UDL, python UDLs need to implement couple functions

- __init__(self,conf_str)
- ocdpo_handler(self,**kwargs)
- __del__(self)

Below are optional functions to implement, needed for python UDLs that have model running on GPUs
- load_model 
- evict_model

## Prerequisites

- apt install python3-dev
- pip install numpy

## Source
This repo is ported from Cascade implementation: https://github.com/Derecho-Project/cascade/tree/master/src/udl_zoo/python