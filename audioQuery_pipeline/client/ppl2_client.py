#!/usr/bin/env python3

import json
import os
import pickle
import sys
import time
from derecho.cascade.external_client import ServiceClientAPI, TimestampLogger

import numpy as np
from pipeline2_serialize_utils import AudioBatcher
from torch.utils.data import DataLoader


BS = 1
NUM_BATCHES = 5
SEND_INTERVAL = 1  # in s
FIRST_UDL_PREFIX = "/speech_to_text/" # micro
FIRST_UDL_PREFIX = "/mono_speech/" # mono
FIRST_UDL_SHARDS = [0,1]
FIRST_UDL_SUBGROUP_TYPE = "VolatileCascadeStoreWithStringKey"
FIRST_UDL_SUBGROUP_INDEX = 0
AUDIO_DIR = "./perf_data/"
AUDIO_PKL_NAME = "queries_audio5000.pkl"


def get_audio_data():
    pkl_dir = os.path.join(AUDIO_DIR, AUDIO_PKL_NAME)
    with open(pkl_dir, "rb") as f:
        waveforms = pickle.load(f)
    list_np_waveform = []
    for i, item in enumerate(waveforms):
        if len(item[-1]) > 200000:
            continue
        list_np_waveform.append(item[-1])
    return list_np_waveform

def collate_numpy(batch):
    # Just return the batch (list of np.ndarrays) without converting to torch.Tensor
    return batch


def main(argv):
    audio_np_list = get_audio_data()
    loader = DataLoader(
        audio_np_list,
        batch_size=BS,
        shuffle=False,
        num_workers=16,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_numpy
    )
    
    print(f"loaded {len(audio_np_list)} audio queries")

    tl = TimestampLogger()
    capi = ServiceClientAPI()
    
    print("Connected to Cascade service ...")
    

    for batch_idx, batch in enumerate(loader):
        if batch_idx > NUM_BATCHES:
            break
        key = FIRST_UDL_PREFIX + str(batch_idx)
        
        audio_query_batch = AudioBatcher()
        audio_query_batch.question_ids = [batch_idx * BS + i for i in range(len(batch))]
        audio_query_batch.audio_data = batch
        query_np = audio_query_batch.serialize()
        # audio_query_batch.print_info()
        
        
        for qid in audio_query_batch.question_ids:
            tl.log(1000, qid, 0, 0)
            
        capi.put_nparray(key, query_np, 
                subgroup_type=FIRST_UDL_SUBGROUP_TYPE,
                subgroup_index=FIRST_UDL_SUBGROUP_INDEX,
                shard_index=FIRST_UDL_SHARDS[batch_idx % len(FIRST_UDL_SHARDS)],
                message_id=batch_idx)
        # print(f"Put key:{key} \n    value:{audio_query_batch.question_ids} to Cascade.")

        if batch_idx == 10:
            time.sleep(20)
            
        time.sleep(SEND_INTERVAL)

    tl.flush("client_timestamp.dat")

if __name__ == "__main__":
    main(sys.argv)
