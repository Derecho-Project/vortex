# rag_demo

## Requirement
1. Derecho : https://github.com/Derecho-Project/derecho.git

2. Cascade : https://github.com/Derecho-Project/cascade.git , with Cascade Python API installed (https://github.com/Derecho-Project/cascade/blob/master/src/service/python/README.md )

3. FAISS : https://github.com/facebookresearch/faiss/blob/main/INSTALL.md 

   NVIDIA toolkit nvcc if run with GPU support

   You can install it via the script in script/install-faiss.sh. (Note in current Vortex, we use FAISS cpp; FAISS python is not needed, and optional to install)


# Configuration
The configuration of the udls could be found in dfgs.json file.

In cfg/dfgs.json.tmp we set the configuration of the UDLs and their dependencies. Some UDL-specific configurations could be set there including "emb_dim", "faiss_search_type", "top_num_centroids". For faiss_search_type, it is a number (0: CPU flat search, 1: GPU flat search, 2: GPU IVF search, ..). Note, configurations such as "emb_dim", "retrieve_docs" are dataset dependent. For different performance testing on different datasets, they are set differently. More details in 4.1 below.

# Run

## Server Commands
In the main branch of this repo, we provide a bare-minimal setup of cascade server and client. It only requires two server nodes to start the service. n0, n1. run ``` cascade_server ``` under the build directory of the corresponding folder starts the service. (Need to build Cascade bin and set ENV Variable PATH correctly to include the directory where cascade_server is installed, ```export PATH=$PATH:${directory of cascade_server}```)


## Client Commands
In this repo of client and server configuration, we use node n2 as client node.

### 1. Object pools
The object pools needed for this pipeline: /rag/emb, /rag/doc, /rag/generate. The configurations of these object pools are in ```/setup/object_pool.list```. The first step of the client node is to create object pools needed by the pipeline.

### 2. put embeddings and documents
Construct the vector database by putting centroids and clusters' embeddings and documents.

#### Vector database Data Storage formats
- embeddings: stored in /rag/emb/* object pools

centroids stored in the format of /rag/emb/centroids_obj/[obj_id], e.g. /rag/emb/centroids_obj/1, /rag/emb/centroids_obj/2

cluster embeddings stored in the format of /rag/emb/clusters/cluster[cluster_id]/[obj_id], e.g. /rag/emb/clusters/cluster1/0, /rag/emb/clusters/cluster2/0

Note that because we use these keys as identifier to the embeddings object, if accidentally put other objects with the same prefix put to Cascade, it could cause unexpected knn search result. 

- documents: stored in cascade as KV objects under /rag/doc object pool in PCSS. Document objects' keys are in the format of [doc_path] = /rag/doc/[doc_identifier]

- embeddings to document path table. To fetch the document on a given embedding from its cluster_id and embedding_id. We keep a table for each cluster. The table matches the embeddings of that cluster to their corresponding pathnames. Using this table, the stored documents could be retrieved as context for the LLM. The tables are stored in Cascade in K/V format, with key as /rag/doc/emb_doc_map/cluster[cluster_id]/[table_id], value is in json with emb_id, document_pathname. There could be more than one table object per cluster, depends on the size of the cluster.



### 3. UDLs
After the vector database is constructed, clients could send batch of queries to cascade service. Queries are triggered by putting KV objects to its first UDL, encode_centroids_search_udl. 

- The key prefix to trigger this udl is /rag/emb/centroids_search/, which defined in /cfg/dfgs.json. After the key prefix, the key could have the identifier for this batch of requests as its suffix. The recommended format is "/rag/emb/centroids_search/client[client_id]_qb[query_batch_id]" (e.g. /rag/emb/centroids_search/client5_qb0). (query_batch_id is not required but used for logging purpose)

- The value is a json ({"query_text": ["query0", "query1"], "query_embedding": [emb0, emb1]}) in bytes formats.




### 4. Performance testing 
Client node is run in node n2. For performance testing, we created client side test scripts under the folder ```/perf_test```. 

#### 4.1. Dataset

The dataset used for perf_test could be download from directory ```/setup/perf_data```. There are two datasets that prepared for experiments.

-  miniset is a subset of MSMARCO dataset(https://microsoft.github.io/msmarco/). It has around 800 documents, and used for a fast-check of the system. You can download it by running ```./download_miniset.sh```

- hotpot15 is from hotpot dataset(https://hotpotqa.github.io). It contains 899,667 documents with 11,5104,539 tokens, and used for performance testing. We have run encoding of these documents to embeddings using text-embedding-3-small from OpenAI API  and knn clustering them into 15 clusters using FAISS. You can download it by running ```./download_hotpot.sh```

- GIST is a dataset for vector database(http://corpus-texmex.irisa.fr). GIST and SIFT are used as a benchmark for FAISS, qbvector, weaviate, ...(https://github.com/tensorflow/datasets/blob/master/docs/catalog/sift1m.md). It contains 1M emebddings with 960 embedding dimensions. It doesn't include the document text with the dataset, so we could only use its embeddings in the system with result in the format of embedding IDs. You can download via ```./download_gist.sh```. We provided a format script to generate the embeddings in the format of the object to put to cascade. You can run by ``` python format_gist.py```

NOTE: More on dfgs.json.tmp settings based on different dataset. 

For miniset, hotpot15, ``` emb_dim: 1024``` for the first two UDLs, and ```retrieve_docs:true``` for the third UDL.

For GIST dataset, ``` emb_dim: 960``` for the first two UDLs, and ```retrieve_docs:false``` for the third UDL.

#### 4.2 Prebuild Indices for HNSW

The dataset should be located under `benchmark/perf_data/<dataset_name>`. The dataset folder should contain
1. `centoids.pkl`
2. `cluster_*.pkl`

If these these two files do not exist as in the case of the gist dataset, run `format_gist.py`.

You can run `./build/build_hnsw_index <path_to_dataset_directory> benchmark/hnsw_index/<dataset_name> -m <space deliminated m values> -e <space deliminated e values>`
to build the indices. The code will prebuild all possible combinations of m and e given.

To configure the cluster search udl to load the correct dataset when using hnsw, configure the `faiss_search_type` to 3 and `dataset_name` to match the name of the folder 
in benchmark/hnsw_index.

```
"user_defined_logic_config_list": [
{
      "emb_dim":1024,
      "top_k":3,
      "faiss_search_type":3,
      "dataset_name": "miniset" // tries to look in benchmark/hnsw_index/miniset
}],
```

Note: the code looks at M, EF_CONSTRUCTION, and EF_SEARCH in `groupped_embeddings_for_search.hpp` to try and load the correct prebuilt index.


#### 4.3. Initialize database
The initialization step is to put the embeddings and documents to store in Cascade and use at query runtime. We provided a scrip that puts centroids' and clusters' embeddings, documents, and embedding-to-document-pathname map into Cascade. 

You can run ```python setup/perf_test_setup.py -p <dataset_directory> -e <embedding_dimension> [-doc]```. The ```<dataset_directory>``` could be either ```perf_data/miniset``` (embedding dimension 1024) or ```perf_data/hotpot15``` (embedding dimension 1024) or ```perf_data/gist``` (embedding dimension 960) depends on the scale of the experiment. For gist dataset, it doesn't contain document, but for the other two, could include -doc in the script to have it also put doc text into cascade .

e.g. ```python setup/perf_test_setup.py -p perf_data/miniset -e 1024 -doc ```

```python setup/perf_test_setup.py -p perf_data/gist -e 960 ```

#### 4.4. Run queries
After initialize the database, you can start to experiment with putting queries to Vortex and get the result. 
- latency experiment client. We wrote a program for testing latency of the pipeline. You can run via  ```./latency_client -n <num_requests> -b <batch_size> -q <dataset_director> -i <interval_between_request> -e <emb_dim>```.  (interval is in us, default emb_dim is 1024) 

e.g. ```./latency_client  -q perf_data/miniset -e 1024 -n <num_requests> -b <batch_size> -i <interval_between_request>```

```./latency_client  -q perf_data/gist -e 960 -n <num_requests> -b <batch_size> -i <interval_between_request>```




## Docker image
We have built a docker image that have nvcc, cascade,derecho, FAISS built and installed. You can pull from the docker image and run it on your environment.

Image name: yy354/rag_dev:v1.0
