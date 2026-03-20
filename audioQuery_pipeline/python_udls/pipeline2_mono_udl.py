#!/usr/bin/env python3
import json
import numpy as np
import os
from typing import Any
import threading
import sys

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from speech_to_text_udl import AudioRecognition
from encode_udl import TextEncoder
from search_doc_udl import SearchRetriever
from text_check_udl import TextChecker
from aggregate_udl import TTSRunner
from pipeline2_serialize_utils import (AudioBatcher, PendingAudioRecDataBatcher)
from workers_util import ExecWorker




class MonoSpeechProcess:
    def __init__(self, conf, parent=None):
        self.parent = parent
        self.device = conf["device"]
        self.speech_model_name = conf["speech_model_name"]
        self.encoder_model_name = conf["encoder_model_name"]
        self.index_dir = conf["index_dir"]
        self.emb_dim = int(conf["emb_dim"])
        self.topk = int(conf["topk"])
        self.doc_dir = conf["doc_dir"]
        self.check_model_name = conf["check_model_name"]
        self.speech_model = AudioRecognition(self.device, self.speech_model_name)
        self.text_encoder = TextEncoder(self.device, self.encoder_model_name)
        self.search_retriever = SearchRetriever(self.device, self.index_dir, self.topk, self.doc_dir)
        self.text_checker = TextChecker(self.device, self.check_model_name)
        self.tts_runner = TTSRunner(self.device)
        self.loaded_models = False
    
    def load_model(self):
        self.speech_model.load_model()
        self.text_encoder.load_model()
        self.search_retriever.load_model()
        self.text_checker.load_model()
        self.tts_runner.load_model()
        self.loaded_models = True
        
    def exec_model(self, pending_audio_batcher: PendingAudioRecDataBatcher):
        if not self.loaded_models:
            self.load_model()
        qids = pending_audio_batcher.question_ids[:pending_audio_batcher.num_pending]
        num_pending = pending_audio_batcher.num_pending
        
        # 1. Speech to text
        for qid in qids:
            self.parent.tl.log(10030, qid, 0, num_pending)
        query_list = self.speech_model.exec_model(pending_audio_batcher.audio_data[:num_pending])
        for qid in qids:
            self.parent.tl.log(10031, qid, 0, num_pending)
            
        # 2. Text encoding
        for qid in qids:
            self.parent.tl.log(20030, qid, 0, num_pending)
        embeddings = self.text_encoder.encoder_exec(query_list)
        for qid in qids:
            self.parent.tl.log(20031, qid, 0, num_pending)
            
        # 3. Search retriever
        for qid in qids:
            self.parent.tl.log(30030, qid, 0, num_pending)
        doc_list = self.search_retriever.search_docs(embeddings)
        for qid in qids:
            self.parent.tl.log(30031, qid, 0, num_pending)
            
        # 4. Text checking
        for qid in qids:
            self.parent.tl.log(50030, qid, 0, num_pending)
        check_result = self.text_checker.docs_check(doc_list)
        for qid in qids:
            self.parent.tl.log(50031, qid, 0, num_pending)
            
        # 5. Audio synthesis
        for qid in qids:
            self.parent.tl.log(60030, qid, 0, num_pending)
        audios = self.tts_runner.model_exec(doc_list)
        for qid in qids:
            self.parent.tl.log(60031, qid, 0, num_pending)
            # self.parent.tl.log(70100, qid, 0, num_pending)
        
        results = {}
        for idx, docs in enumerate(doc_list):
            results[qids[idx]] = {
                "docs": docs,
                "check_result": check_result[idx],
                "audios": audios[idx]
            }
            # print(f"Mono Speech PPL2 UDL: {qids[idx]} docs: {docs} check_result: {check_result[idx]} ")
        for qid in qids:
            if qid == self.parent.flush_id:
                print(f"Mono Speech PPL2 UDL: {qid} flush_id")
        return results
        


class MonoSpeechModelWorker(ExecWorker):
    '''
    This is a batch executor for faiss searcher execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.mono_speech_process = MonoSpeechProcess(self.parent.conf, self.parent)
    
    def create_pending_manager(self):
        return PendingAudioRecDataBatcher(self.max_exe_batch_size)


    def main_loop(self):
        batch = None
        while self.running:
            if not batch is None:
                batch.reset()
            with self.cv:
                self.current_batch = -1
                if self.pending_batches[self.next_to_process].num_pending == 0:
                    self.cv.wait(timeout=self.batch_time_us/1000000)   
                if self.pending_batches[self.next_to_process].num_pending != 0:
                    self.current_batch = self.next_to_process
                    self.next_to_process = (self.next_to_process + 1) % len(self.pending_batches)
                    batch = self.pending_batches[self.current_batch]

                    if self.current_batch == self.next_batch:
                        self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
                    self.new_space_available = True
                    self.cv.notify()
            if not self.running:
                break
            if self.current_batch == -1 or not batch:
                continue
            
            # Execute the batch
            results = self.mono_speech_process.exec_model(batch)
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(70100, qid, 0, batch.num_pending)
                

class MonoSpeechUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        

        # Note: batch size here is topk(5) * max_exe_batch_size, 
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        self.flush_id = self.conf.get("flush_id", 0)
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = MonoSpeechModelWorker(self, 1)
            self.model_worker.start()

    def ocdpo_handler(self, **kwargs):
        # Only start the model_worker if this UDL is triggered on this node
        if not self.model_worker:
            self.start_threads()
        data = kwargs["blob"]
        
        audio_batcher = AudioBatcher()
        audio_batcher.deserialize(data)
        for qid in audio_batcher.question_ids:
            self.tl.log(10000, qid, 0, 0)
        self.model_worker.push_to_pending_batches(audio_batcher)


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Mono Speech PPL2 UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()