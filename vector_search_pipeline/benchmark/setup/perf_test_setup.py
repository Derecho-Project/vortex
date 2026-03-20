#!/usr/bin/env python3

import numpy as np
import os
import pickle
import sys
import time
import json
import argparse
from derecho.cascade.external_client import ServiceClientAPI



NUM_EMB_PER_OBJ = 200  # < 1MB/4KB = 250 (suppose p2p message size is 1MB)
NUM_KEY_PER_MAP_OBJ = 50000 # takes around 1MB memory
FLOAT_POINT_SIZE = 32  # currently only support 32-bit float TODO: add support for 64-bit float

np.random.seed(1234)             # make reproducible
script_dir = os.path.dirname(os.path.abspath(__file__))

OBJECT_POOLS_LIST = "object_pools.list"

DOC_EMB_MAP_FILENAME = "doc_emb_map.pkl"
DOC_LIST_FILENAME = "doc_list.pkl"
CENTROIDS_FILENAME = "centroids.pkl"
CLUSTER_FILE_PREFIX = "cluster_"


SUBGROUP_TYPES = {
        "VCSS": "VolatileCascadeStoreWithStringKey",
        "PCSS": "PersistentCascadeStoreWithStringKey",
        "TCSS": "TriggerCascadeNoStoreWithStringKey"
        }




def create_object_pool(capi, basepath):
    # create object pools
    print("Creating object pools ...")
    fpath = os.path.join(basepath,OBJECT_POOLS_LIST)
    with open(fpath,"r") as list_file:
        for line in list_file:
            fields = line.strip().split(" ")
            pool_path = fields[0]
            subgroup_type = fields[1]
            subgroup_index = int(fields[2])
            if len(fields) >= 4:
                affinity_set_regex = fields[3].strip()
                res = capi.create_object_pool(pool_path,SUBGROUP_TYPES[subgroup_type],
                                              subgroup_index,
                                              affinity_set_regex=affinity_set_regex)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
                    exit(1)
            else:
                print(f"  {pool_path} {subgroup_type} {subgroup_index}")
                res = capi.create_object_pool(pool_path,
                                              SUBGROUP_TYPES[subgroup_type],
                                              subgroup_index)
                if res:
                    res.get_result()
                else:
                    print(f"Failed to create the object pool: {pool_path}")
                    exit(1)


def get_embeddings(basepath, filename, d=1024):
    '''
    Load the embeddings from a pickle file.
    '''
    fpath = os.path.join(basepath, filename)
    with open(fpath, 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, list):
        data = np.array(data)
    if data.dtype == np.float64 and FLOAT_POINT_SIZE == 32:
        float32_data = data.astype(np.float32)
        data = float32_data
    assert(data.shape[1] == d)
    return data


def break_into_chunks(num_embeddings, chunk_size):
    chunk_idxs = []
    num_chunks = num_embeddings // chunk_size + 1 if num_embeddings % chunk_size != 0 else num_embeddings // chunk_size
    remaining_embs_num = num_embeddings
    for i in range(num_chunks):
        start_idx = i*chunk_size
        end_idx = i*chunk_size + min(chunk_size, remaining_embs_num)
        chunk_idxs.append((start_idx, end_idx))
        remaining_embs_num -= chunk_size
    return chunk_idxs

    
def put_initial_embeddings_docs(capi, basepath, put_docs=True, embed_dim=1024):
    # 0. put answer mapping
    DOC_EMB_MAP = pickle.load(open(os.path.join(basepath, DOC_EMB_MAP_FILENAME), "rb"))
    print("Initializing: putting doc_emb map to cascade server ...")
    for i, (cluster_id, table_dict) in enumerate(DOC_EMB_MAP.items()):
        # break the table into chunks
        chunk_idx = break_into_chunks(len(table_dict), NUM_KEY_PER_MAP_OBJ)
        table_key_list = list(table_dict.keys())
        for j, (start_idx, end_idx) in enumerate(chunk_idx):
            key = f"/rag/doc/emb_doc_map/cluster{cluster_id}/{j}"
            table_dict_chunk = {k: table_dict[k] for k in table_key_list[start_idx:end_idx]}
            table_json = json.dumps(table_dict_chunk)
            res = capi.put(key, table_json.encode('utf-8'))
            if res:
                res.get_result()
            else:
                print(f"Failed to put the doc table to key: {key}")
                exit(1)
        print(f"         Put cluster{cluster_id} doc_emb_map size: {len(table_dict)}")
    # 1. Put centroids'embeddings to cascade.
    centroids_embs = get_embeddings(basepath, CENTROIDS_FILENAME, embed_dim)
    centroids_chunk_idx = break_into_chunks(centroids_embs.shape[0], NUM_EMB_PER_OBJ)
    print(f"Initilizing: put {centroids_embs.shape[0]} centroids embeddings to cascade")
    for i, (start_idx, end_idx) in enumerate(centroids_chunk_idx):
        key = f"/rag/emb/centroids_obj/{i}"
        centroids_embs_chunk = centroids_embs[start_idx:end_idx]
        res = capi.put(key, centroids_embs_chunk.tobytes())
        if res:
            res.get_result()
        else:
            print(f"Failed to put the centroids embeddings to key: {key}")
            exit(1)
    
    
    # 2. Put clusters' embeddings and docs to cascade.
    if put_docs:
        doc_list = pickle.load(open(os.path.join(basepath, DOC_LIST_FILENAME), "rb"))
        print("Initializing: putting clusters' embeddings and docs to cascade server ...")
    else:
        print("Initializing: putting clusters' embeddings to cascade server ...")

    centroid_count = centroids_embs.shape[0]
    cluster_file_name_list = [f'{CLUSTER_FILE_PREFIX}{count}.pkl' for count in range(centroid_count)]
    for cluster_id, cluster_file_name in enumerate(cluster_file_name_list):
        cluster_embs = get_embeddings(basepath, cluster_file_name, embed_dim)
        num_embeddings = cluster_embs.shape[0]
        cluster_chunk_idx = break_into_chunks(num_embeddings, NUM_EMB_PER_OBJ)
        for i, (start_idx, end_idx) in enumerate(cluster_chunk_idx):
            key = f"/rag/emb/clusters/cluster{cluster_id}/{i}"
            cluster_embs_chunk = cluster_embs[start_idx:end_idx]
            res = capi.put(key, cluster_embs_chunk.tobytes())
            if res:
                res.get_result()
            else:
                print(f"Failed to put the cluster embeddings to key: {key}")
                exit(1)
        # Put the corresponding docs to cascade
        if put_docs:
            for emb_idx in range(num_embeddings):
                doc_idx = DOC_EMB_MAP[cluster_id][emb_idx]
                doc_key = f"/rag/doc/{doc_idx}"
                doc = doc_list[doc_idx]
                res = capi.put(doc_key, doc.encode('utf-8'))
                if res:
                    res.get_result()
                else:
                    print(f"Failed to put the doc to key: {doc_key}")
                    exit(1)
        print(f"         Put cluster{cluster_id}, num {num_embeddings} embs, {len(cluster_chunk_idx)} objs to cascade")
    print(f"Initialized embeddings")


    


def main(argv):
    print("Connecting to Cascade service ...")   

    parser = argparse.ArgumentParser(description="Set up and connect to the Cascade service.")
    parser.add_argument('-p', '--path', required=True, type=str, help="Path to the data folder.")
    parser.add_argument('-e', '--embed_dim', required=True, type=int, help="Dimension of embeddings.")
    parser.add_argument('-doc', action='store_true', help="Include docs in the setup.")

    # Parse arguments
    args = parser.parse_args()

    # Access parsed arguments
    data_dir = args.path
    include_docs = args.doc
    embed_dim = args.embed_dim

    # Connect to Cascade service
    print("Connecting to Cascade service ...")
    capi = ServiceClientAPI()
    create_object_pool(capi, script_dir)
    put_initial_embeddings_docs(capi, data_dir, put_docs=include_docs, embed_dim=embed_dim)
    print("Done!")





if __name__ == "__main__":
    main(sys.argv)
