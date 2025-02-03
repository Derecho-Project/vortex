#!/usr/bin/env python3
import json
import struct
import warnings

import numpy as np

from typing import Any
from FlagEmbedding import BGEM3FlagModel

import cascade_context
from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import TimestampLogger

warnings.filterwarnings("ignore")

class Batch:
    def __init__(self, capacity: int):
        self._size = 0 
        self._capacity = capacity

        self._query_ids = np.ndarray(shape=(self._capacity, ), dtype=np.uint64)
        self._client_ids = np.ndarray(shape=(self._capacity, ), dtype=np.uint32)
        self._text: list[str] = [""] * self._capacity
        self._total_text_size = 0
    
    @property
    def query_list(self):
        return self._text[:self._size]

    def deserialize(self, data: bytes):
        # NOTE: assumes the endiannesss of machines is same
        # reference a slice of the data without copying it
        view = memoryview(data)
        offset = 0

        # unpack how many records
        (count,) = struct.unpack_from('<I', data, offset)
        offset += 4

        self._total_text_size = 0
        self._size = count
        self.resize(count)

        record_format = "<QII" # Q = 8 bytes, I = 4 bytes
        record_size = struct.calcsize(record_format)
        for idx in range(count):
            # unpack query id, client id, and string length
            query_id, client_id, str_len = struct.unpack_from(record_format, view, offset)
            offset += record_size

            # unpack string
            str_view = view[offset:offset+str_len] 
            offset += str_len
            str_val = str_view.tobytes().decode('utf-8')
            self._total_text_size += str_len

            # save
            self._query_ids[idx] = query_id
            self._client_ids[idx] = client_id
            self._text[idx] = str_val 

    def serialize(self, embeddings: np.ndarray) -> bytes:
        # I'm sorry to who ever is reading this, but this is extrememly suspicious
        # mimic the byte layout of EmbeddingQueryBatcher

        _, emb_dims = embeddings.shape
        
        # num queries, embeddings position
        header_format = "<II"
        header_size = struct.calcsize(header_format)

        # query id, client id, text position, text length, embedding position, emb length
        metadata_format = "<QIIIII"
        metadata_size = struct.calcsize(metadata_format)

        metadata_position = header_size
        text_position = metadata_position + (self._size * metadata_size)
        embeddings_position = text_position + self._total_text_size

        # pack header
        data = bytearray()
        data += struct.pack(header_format, self._size, embeddings_position)

        # pack metadata
        for i in range(self._size):
            query_id = self._query_ids[i]
            client_id = self._client_ids[i]
            text_pos = text_position
            text_len = len(self._text[i])
            emb_pos = embeddings_position + i * emb_dims
            emb_len = emb_dims

            text_position += text_len

            data += struct.pack(metadata_format, query_id, client_id, text_pos, text_len, emb_pos, emb_len)
        
        # pack string
        for s in self._text:
            data += s.encode('utf-8')

        for i in range(self._size):
            data += embeddings[i, :].tobytes(order='C')

        return data
    
    def resize(self, new_size: int):
        """resize batch manager if needed"""
        if new_size > self._capacity:
            self._capacity = new_size 
            self._query_ids = np.resize(self._query_ids, new_shape=(self._capacity, ))
            self._client_ids = np.resize(self._query_ids, new_shape=(self._capacity, ))
            self._text = [""] * self._capacity

class EncodeSearchUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):

        conf: dict[str, Any] = json.loads(conf_str)

        self._tl = TimestampLogger()
        self._encoder = BGEM3FlagModel(
            model_name_or_path=conf["encoder_config"]["model"],
            device=conf["encoder_config"]["device"],
            user_fp16=False,
        )
        self._emb_dim = int(conf["emb_dim"])
        self._batch = Batch(64)
        self._batch_id = 0

    def ocdpo_handler(self, **kwargs):
        key = kwargs["key"]
        bytes = kwargs["blob"].tobytes()

        self._batch.deserialize(bytes)
        
        res: Any = self._encoder.encode(
            self._batch.query_list,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        query_embeddings: np.ndarray = res["dense_vecs"]
        query_embeddings_trunc = query_embeddings[:, :self._emb_dim]
        print("#########################")
        print("key:", key)
        print("query_list: ", query_embeddings_trunc.shape)
        
        self._batch_id += 1
        key_str = f"/rag/emb/centroids_search/batch{self._batch_id}_"
        cascade_context.emit(key, self._batch.serialize(query_embeddings_trunc))
    
    def __del__(self):
        pass
