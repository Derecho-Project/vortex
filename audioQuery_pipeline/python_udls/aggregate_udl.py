#!/usr/bin/env python3
import logging
logging.disable(logging.CRITICAL)

import warnings
import json
import numpy as np
from typing import Any
import threading
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from torch.nn.utils.rnn import pad_sequence

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

from pipeline2_serialize_utils import (DocResultBatchManager, 
                                       TextCheckResultBatchManager,
                                       PendingCheckDataBatcher)
from workers_util import ExecWorker


class TTSRunner:
    def __init__(self, device: str):
        self.fastpitch = None
        self.FASTPITCH_NAME = "nvidia/tts_en_fastpitch"
        self.hifigan = None
        self.HIFIGAN_NAME = "nvidia/tts_hifigan"
        self.device = device
        
    def load_model(self):
        self.fastpitch = FastPitchModel.from_pretrained(self.FASTPITCH_NAME).to(self.device).eval()
        self.hifigan = HifiGanModel.from_pretrained(model_name=self.HIFIGAN_NAME).to(self.device).eval()
        print("TTS model loaded")
        
    def run_tts(self, batch_texts: list[str]) -> np.ndarray:
        if self.fastpitch is None:
            self.load_model()
        token_list = [self.fastpitch.parse(text).squeeze(0) for text in batch_texts]
        tokens = pad_sequence(token_list, batch_first=True).to(self.device)
        with torch.no_grad():
            spectrograms = self.fastpitch.generate_spectrogram(tokens=tokens)
            audios = self.hifigan.convert_spectrogram_to_audio(spec=spectrograms)
        np_audios = audios.cpu().numpy()
        return np_audios
    
    def model_exec(self, batch_docs:list[list[str]]) -> list[list[np.ndarray]]:
        flattened_doc_list = [item for sublist in batch_docs for item in sublist]
        tts_audios = self.run_tts(flattened_doc_list)
        # Reshape the audios to match the original doc_list structure
        reshaped_audios = []
        start = 0
        for sublist in batch_docs:
            end = start + len(sublist)
            reshaped_audios.append(tts_audios[start:end])
            start = end
        return reshaped_audios

    
class TTSModelWorker(ExecWorker):
    '''
    This is a batch executor for TTS model execution
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.tts_model = TTSRunner(self.parent.device)
        

    def create_pending_manager(self):
        return PendingCheckDataBatcher(self.max_exe_batch_size)
    

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
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(70030, qid, 0, batch.num_pending)
                
            audio_lists = self.tts_model.model_exec(batch.doc_list[:batch.num_pending])
            
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(70031, qid, 0, batch.num_pending)
                
            self.parent.collector.add_doc_result(batch,
                                        audio_lists, 
                                        batch.num_pending)
            self.pending_batches[self.current_batch].reset()


class CollectedResult:
    def __init__(self, question_id):
        self.question_id = question_id
        self.question_text = None
        self.doc_list = None
        self.text_check_list = None  # a list of doc_type IDs: 0 hate speech, 1 offensive language, 2 neither
        self.audio_list = None

        
    def collect_all(self):
        if self.question_text is None:
            return False
        if self.doc_list is None:
            return False
        if self.text_check_list is None:
            return False
        if self.audio_list is None:
            return False
        return True
    
    def print_result(self):
        print(f"Question ID: {self.question_id}")
        print(f"Question Text: {self.question_text}")
        print(f"Document List: {self.doc_list}")
        print(f"Text Check List: {self.text_check_list}")
        print(f"Audio List: {len(self.audio_list)}, size: {self.audio_list[0].shape}")


class Collector:
    def __init__(self, tl: TimestampLogger, flush_id: str):
        self.collected_results = {}
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.finished_count = 0
        self.tl = tl
        self.flush_id = flush_id

    def check_collected_result(self, qid):
        '''
        Not acquiring the lock, since it is only called by add_to_collection
        '''
        if self.collected_results[qid].collect_all():
            self.finished_count += 1
            # self.collected_results[qid].print_result()
            
            del self.collected_results[qid]
            self.tl.log(70100, qid, 0, 1)
            if qid == self.flush_id:
                print(f"AGG received all results for question ID {qid}")
                # self.collected_results[qid].print_result()
                print(f"AGG finished count: {self.finished_count}")
    

    def add_doc_result(self,doc_result_batch: PendingCheckDataBatcher, audio_list: list[list[np.ndarray]], num_queries):
        with self.cv:
            for idx, qid in enumerate(doc_result_batch.question_ids[:num_queries]):
                
                if qid not in self.collected_results:
                    self.collected_results[qid] = CollectedResult(qid)
                self.collected_results[qid].question_text = doc_result_batch.queries[idx]
                self.collected_results[qid].doc_list = doc_result_batch.doc_list[idx]#.copy()
                self.collected_results[qid].audio_list = audio_list[idx]#.copy()
                if len(self.collected_results[qid].doc_list) == 0:
                    warnings.warn(f"AGG received an empty doc list for question ID {qid}")
                self.check_collected_result(qid)
                
            self.cv.notify_all()


    def add_text_check_result(self, text_check_result_batch: TextCheckResultBatchManager):
        with self.cv:
            for idx, qid in enumerate(text_check_result_batch.question_ids):
                self.tl.log(70010, qid, 0, 0)
                
                if qid not in self.collected_results:
                    self.collected_results[qid] = CollectedResult(qid)
                self.collected_results[qid].text_check_list = text_check_result_batch.doc_types[idx].copy()
                if len(self.collected_results[qid].text_check_list) == 0:
                    warnings.warn(f"AGG received an empty text check result for question ID {qid}")
                    
                self.check_collected_result(qid)
                    
            self.cv.notify_all()


class AggregateUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        # self.collected_results = {}
        self.finished_count = 0
        
        self.model_worker = None
        self.device = self.conf["device"]
        self.collector = Collector(self.tl, self.conf["flush_id"])
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        self.flush_id = self.conf["flush_id"]
    

    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = TTSModelWorker(self, 1)
            self.model_worker.start()

    def ocdpo_handler(self, **kwargs):
        data = kwargs["blob"]
        key = kwargs["key"]
        if not self.model_worker:
            self.start_threads()
        
        if key.find("doc") != -1:
            # If it were doc, then first put it to the TTS execution queue 
            doc_batcher = DocResultBatchManager()
            doc_batcher.deserialize(data)
            for qid in doc_batcher.question_ids:
                self.tl.log(70001, qid, 0, 0)
            # Data content is copied once when forming batch at add_data
            self.model_worker.push_to_pending_batches(doc_batcher)
        
        if key.find("check") != -1:
            # text check result
            check_result_batcher = TextCheckResultBatchManager()
            check_result_batcher.deserialize(data.copy())
            self.collector.add_text_check_result(check_result_batcher)
            
            
            
        


    def __del__(self):
        '''
        Destructor
        '''
        print(f"Agg UDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
