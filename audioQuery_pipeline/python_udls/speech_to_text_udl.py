#!/usr/bin/env python3
import json
import numpy as np
import os
from typing import Any
import threading
import sys

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import extract_fbank

from derecho.cascade.udl import UserDefinedLogic
from derecho.cascade.member_client import ServiceClientAPI
from derecho.cascade.member_client import TimestampLogger

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SenseVoice"))
from SenseVoice.utils.frontend import WavFrontend, WavFrontendOnline
from SenseVoice.model import SenseVoiceSmall

from workers_util import ExecWorker, EmitWorker
from pipeline2_serialize_utils import AudioBatcher, PendingAudioRecDataBatcher, QueryBatcherManager


AUDIO_NEXT_UDL_PREFIX = "/encode_search/"
AUDIO_NEXT_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
AUDIO_NEXT_UDL_SUBGROUP_INDEX = 0

class AudioRecognition:
    def __init__(self, device_name, model_dir, language="en"):
        '''
        language: "zh", "en", "yue", "ja", "ko", "nospeech","auto"
        '''
        self.language = language
        self.model_dir = model_dir # "iic/SenseVoiceSmall"
        self.model = None
        self.device_name = device_name
        self.kwargs = None
        self.frontend = None

    def load_model(self):
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(model=self.model_dir, device=self.device_name)
        self.model.eval()
        self.kwargs["data_type"] = "fbank"
        self.kwargs["sound"] = "fbank"
        self.frontend = self.kwargs["frontend"]
        print("Speech to Text model loaded")
        
    def exec_model(self, batch_audios):
        if self.model is None:
            self.load_model()
        speech, speech_lengths = extract_fbank(
            batch_audios, data_type=self.kwargs.get("data_type", "sound"), frontend=self.frontend
        )
        res = self.model.inference(
            data_in=speech,
            data_lengths=speech_lengths,
            language=self.language, 
            use_itn=False,
            ban_emo_unk=True,
            **self.kwargs,
        )
        text_list = []
        for idx in range(len(res[0])):
            text_list.append(rich_transcription_postprocess(res[0][idx]["text"]))
        return text_list

    # def exec_model(self, batch_audios):
    #     if self.model is None:
    #         self.load_model()

    #     # Step 1: Separate audios by length, while preserving their original indices
    #     short_audios = []
    #     long_audio_tasks = []
    #     for idx, audio in enumerate(batch_audios):
    #         if len(audio) > 200000:
    #             long_audio_tasks.append((idx, audio))
    #         else:
    #             short_audios.append((idx, audio))

    #     # Step 2: Prepare result list
    #     text_list = [None] * len(batch_audios)

    #     # Step 3: Run batched inference on short audios
    #     if short_audios:
    #         indices, short_audio_data = zip(*short_audios)
    #         speech, speech_lengths = extract_fbank(
    #             list(short_audio_data), 
    #             data_type=self.kwargs.get("data_type", "sound"), 
    #             frontend=self.frontend
    #         )
    #         res = self.model.inference(
    #             data_in=speech,
    #             data_lengths=speech_lengths,
    #             language=self.language,
    #             use_itn=False,
    #             ban_emo_unk=True,
    #             **self.kwargs,
    #         )
    #         for i, idx in enumerate(indices):
    #             text_list[idx] = rich_transcription_postprocess(res[0][i]["text"])

    #     # Step 4: Run individual inference on long audios
    #     for idx, audio in long_audio_tasks:
    #         speech, speech_lengths = extract_fbank(
    #             [audio], 
    #             data_type=self.kwargs.get("data_type", "sound"), 
    #             frontend=self.frontend
    #         )
    #         res = self.model.inference(
    #             data_in=speech,
    #             data_lengths=speech_lengths,
    #             language=self.language,
    #             use_itn=False,
    #             ban_emo_unk=True,
    #             **self.kwargs,
    #         )
    #         text_list[idx] = rich_transcription_postprocess(res[0][0]["text"])

    #     return text_list


class SpeechTextWorker(ExecWorker):
    '''
    This is a batch executor for the audio recognition model
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        self.max_exe_batch_size = self.parent.max_exe_batch_size
        self.batch_time_us = self.parent.batch_time_us
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.audio_rec_model = AudioRecognition(self.parent.device, self.parent.model_name)

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
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(10030, qid, 0, batch.num_pending)
            
            query_list = self.audio_rec_model.exec_model(batch.audio_data[:batch.num_pending])
            for qid in batch.question_ids[:batch.num_pending]:
                self.parent.tl.log(10031, qid, 0, batch.num_pending)
            self.parent.emit_worker.add_to_buffer(batch,
                                                  query_list, 
                                                  batch.num_pending)
            self.pending_batches[self.current_batch].reset()
            
    # def push_to_pending_batches(self, queryBatcher):
    #     num_questions = len(queryBatcher.question_ids)
    #     question_to_add_id = 0
    #     # Add the data one-by-one
    #     while question_to_add_id < num_questions:
    #         with self.cv:
    #             while True:
    #                 if not self.new_space_available:
    #                     self.cv.wait(timeout=3)
    #                 if not self.running:
    #                     break
    #                 if self.new_space_available:
    #                     break
    #             cur_audio_len = queryBatcher.audio_data[question_to_add_id].shape[0]
    #             free_batch = self.next_batch
    #             space_left = self.pending_batches[free_batch].space_left_large(cur_arr_len=cur_audio_len)
    #             initial_batch = free_batch
    #             # Find the idx in the pending_batches to add the data
    #             while space_left == 0:
    #                 free_batch = (free_batch + 1) % len(self.pending_batches)
    #                 if free_batch == self.current_batch:
    #                     free_batch = (free_batch + 1) % len(self.pending_batches)
    #                 if free_batch == initial_batch:
    #                     break
    #                 space_left = self.pending_batches[free_batch].space_left_large(cur_arr_len=cur_audio_len)
    #             if space_left != 0:
    #                 # add as many questions as possible to the pending batch
    #                 self.next_batch = free_batch
    #                 question_start_idx = question_to_add_id
    #                 end_idx = self.pending_batches[free_batch].add_single_data(queryBatcher, question_start_idx)
    #                 question_to_add_id = end_idx
    #                 #  if we complete filled the buffer, cycle to the next
    #                 if self.pending_batches[free_batch].space_left_large(cur_arr_len=cur_audio_len) == 0:
    #                     self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
    #                     if self.next_batch == self.current_batch:
    #                         self.next_batch = (self.next_batch + 1) % len(self.pending_batches)
    #                 self.cv.notify()
    #             else:
    #                 self.new_space_available = False


class SpeechTextEmitWorker(EmitWorker):
    '''
    This is a batcher for SpeechToTextUDL to emit to centroids search UDL
    '''
    def __init__(self, parent, thread_id):
        super().__init__(parent, thread_id)
        
        self.emit_log_flag = 10100
        self.max_emit_batch_size = self.parent.max_emit_batch_size
        self.initial_pending_batch_num = self.parent.num_pending_buffer
        self.next_udl_subgroup_type = AUDIO_NEXT_UDL_SUBGROUP_TYPE
        self.next_udl_subgroup_index = AUDIO_NEXT_UDL_SUBGROUP_INDEX
        self.next_udl_shards = self.parent.next_udl_shards
        self.next_udl_prefix = AUDIO_NEXT_UDL_PREFIX
    
    def create_batch_manager(self):
        # Return an instance of the batch manager that this child class needs.
        return QueryBatcherManager()

    def add_to_buffer(self, batch, query_list, num_queries):
        '''
        pass by object reference to avoid deep-copy
        '''
        question_ids = batch.question_ids[:num_queries]
        
        with self.cv:
            # use question_id to determine which shard to send to
            for i in range(num_queries):
                shard_pos = question_ids[i] % len(self.parent.next_udl_shards)
                self.send_buffer[shard_pos].add_result(question_ids[i],
                                                       query_list[i])
            self.cv.notify()


class SpeechToTextUDL(UserDefinedLogic):
    def __init__(self, conf_str: str):
        self.conf: dict[str, Any] = json.loads(conf_str)
        self.capi = ServiceClientAPI()
        self.tl = TimestampLogger()
        self.device = self.conf["device"]
        self.model_name = self.conf["model_name"]
        self.max_exe_batch_size = int(self.conf.get("max_exe_batch_size", 16))
        self.batch_time_us = int(self.conf.get("batch_time_us", 1000))
        self.max_emit_batch_size = int(self.conf.get("max_emit_batch_size", 5))
        self.next_udl_shards = self.conf.get("next_udl_shards", [0,1])
        self.num_pending_buffer = self.conf.get("num_pending_buffer", 10)
        
        self.model_worker = None
        self.emit_worker = None
        self.sent_msg_count = 0
        
    def start_threads(self):
        '''
        Start the worker threads
        '''
        if not self.model_worker:
            self.model_worker = SpeechTextWorker(self, 1)
            self.model_worker.start()
            self.emit_worker = SpeechTextEmitWorker(self, 2)
            self.emit_worker.start()

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
        print(f"SpeechToTextUDL destructor")
        if self.model_worker:
            self.model_worker.signal_stop()
            self.model_worker.join()
        if self.emit_worker:
            self.emit_worker.signal_stop()
            self.emit_worker.join()
