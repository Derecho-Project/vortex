#!/usr/bin/env python3
import time
import json
import warnings

import numpy as np

from typing import Any
from FlagEmbedding import BGEM3FlagModel

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger

warnings.filterwarnings("ignore")

class Batch:
    def __init__(self):
        self._bytes: np.ndarray = np.ndarray(shape=(0, ), dtype=np.uint8)
        self._strings: list[str] = []
        self._client_id: int = 0
    
    @property
    def query_list(self):
        return self._strings
    
    @property
    def client_id(self):
        return self._client_id

    def deserialize(self, data: np.ndarray):
        self._bytes = data

        # structured dtype
        header_type = np.dtype([
            ('count', np.uint32),
            ('embeddings_start', np.uint32)
        ])
        metadata_type = np.dtype([
            ('query_id', np.uint64),
            ('client_id', np.uint32),
            ('text_position', np.uint32),
            ('text_length', np.uint32),
            ('embeddings_position', np.uint32),
            ('embeddings_dim', np.uint32),
        ])

        header_start = 0
        header_end = header_start + header_type.itemsize
        (count, _) = data[header_start:header_end].view(header_type)[0]

        metadata_start = 8
        metadata_end = metadata_type.itemsize * count + metadata_start
        self._strings = [""] * count

        # get one record to grab client id
        # saves on bne in loop
        metadata_record = data[metadata_start:metadata_start + metadata_type.itemsize].view(metadata_type)[0]
        self._client_id = metadata_record[1]

        for idx, m in enumerate(data[metadata_start:metadata_end].view(metadata_type)):
            string_start = m[2]
            string_length = m[3]
            string = data[string_start:string_start+string_length].tobytes().decode("utf-8")
            self._strings[idx] = string


    def serialize(self, embeddings: np.ndarray) -> bytes:
        # lesson learned
        # do not use .astype as that is a cast on each element of the array
        # use .view, which is simular to C++'s reinterpret_cast
        return np.concatenate((self._bytes, embeddings.flatten().view(np.uint8))).tobytes()
    
class EncodeSearchUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):

        conf: dict[str, Any] = json.loads(conf_str)

        self._tl = TimestampLogger()
        self._encoder = BGEM3FlagModel(
            model_name_or_path=conf["encoder_config"]["model"],
            device=conf["encoder_config"]["device"],
            use_fp16=False,
        )
        self._emb_dim = int(conf["emb_dim"])
        self._batch = Batch()
        self._batch_id = 0

    def ocdpo_handler(self, **kwargs):
        data = kwargs["blob"]
        
        self._batch.deserialize(data)
        
        res: Any = self._encoder.encode(
            self._batch.query_list,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embeddings: np.ndarray = res["dense_vecs"]
        query_embeddings_trunc = query_embeddings[:, :self._emb_dim]
        self._batch_id += 1


        # format should be {client}_{batch_id}
        key_str = f"{self._batch.client_id}_{self._batch_id}"
        output_bytes = self._batch.serialize(query_embeddings_trunc)
        cascade_context.emit(key_str, output_bytes, message_id=kwargs["message_id"])
        return None
    
    def __del__(self):
        pass
