#pragma once

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <hnswlib/hnswlib.h>

#include "serialize_utils.hpp"

#define MAX_NUM_QUERIES_PER_BATCH 1000

namespace derecho{
namespace cascade{

constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * This struct is used to store HNSW configuration used to either load prebuilt indicies
 * or build indicies during runtime.
 */
struct HNSWConfig {
    std::string dataset_name {""}; // name of the dataset, used by hnsw to determine which prebuilt index to load
    int m {48};                    // controls how dense the graph is, higher --> denser
    int ef_construction {200};     // how many neighbors are checked when adding a new point (should be > top_k)
    int ef_search {200};           // how many neighbors are explored to find the closest match  (should be > top_k)
};

/*
 * This class is responsible for holding a batch of queries to be processed in a memory-efficient way.
 */
class PendingEmbeddingQueryBatch {
    uint64_t emb_dim;
    float* embeddings;
    uint64_t max_queries = 0;
    uint64_t num_queries = 0;
    std::vector<std::shared_ptr<EmbeddingQuery>> queries;

public:
    PendingEmbeddingQueryBatch(uint64_t emb_dim,uint64_t max_size);
    ~PendingEmbeddingQueryBatch();
    
    uint64_t add_queries(const std::vector<std::shared_ptr<EmbeddingQuery>>& queries,
        uint64_t query_start_index,
        uint64_t num_to_add,
        const uint8_t *buffer,
        uint32_t embeddings_position,
        uint32_t embeddings_size);

    const float * get_embeddings();
    const std::vector<std::shared_ptr<EmbeddingQuery>>& get_queries();
    uint64_t capacity();
    uint64_t size();
    uint64_t space_left();
    bool empty();
    void reset();
};

/*** Wrapper for the ANN search engine
 * Store group of embeddings for a cluster or for centroids;
 * Build the index for the embeddings
 */
class GroupedEmbeddingsForSearch{
public:
     enum class SearchType {
          FaissCpuFlatSearch = 0,
          FaissGpuFlatSearch = 1,
          FaissGpuIvfSearch = 2,
          HnswlibCpuSearch = 3
     };

     SearchType search_type; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
     int emb_dim;  //  e.g. 512. The dimension of each embedding
     int num_embs;  //  e.g. 1000. The number of embeddings in the array

     // hnsw hyperparamters: looks at `hnsw_m` `
     // hnswlib looks at these parameters to figure out which prebuilt index to load if any,
     // or what hyperparameters are used to build/search
     const HNSWConfig hnsw_config;

     float* embeddings; 
     std::atomic<bool> initialized_index; // Whether the index has been built

     std::unique_ptr<faiss::IndexFlatL2> cpu_flatl2_index; // FAISS index object. Initialize if use CPU Flat search
     std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;  // FAISS GPU resources. Initialize if use GPU search
     std::unique_ptr<faiss::gpu::GpuIndexFlatL2> gpu_flatl2_index; // FAISS index object. Initialize if use GPU Flat search
     std::unique_ptr<faiss::gpu::GpuIndexIVFFlat> gpu_ivf_flatl2_index; // FAISS index object. Initialize if use GPU IVF search
     std::unique_ptr<hnswlib::HierarchicalNSW<float>> cpu_hnsw_index; // HNSWLIB index object. Initialize if use hnswlib cpu search 
     std::unique_ptr<hnswlib::L2Space> l2_space;



     GroupedEmbeddingsForSearch(int type, int dim, HNSWConfig hnsw_config)
          :search_type(static_cast<SearchType>(type)), emb_dim(dim), num_embs(0), embeddings(nullptr), hnsw_config(hnsw_config) {
          initialized_index.store(false);
     }

     GroupedEmbeddingsForSearch(int type, int dim) 
          : search_type(static_cast<SearchType>(type)), emb_dim(dim), num_embs(0), embeddings(nullptr) {
          initialized_index.store(false);
     }

     GroupedEmbeddingsForSearch(int dim, int num, float* data) 
          : search_type(SearchType::FaissCpuFlatSearch), emb_dim(dim), num_embs(num), embeddings(data){
          initialized_index.store(false);
     }

     /*** 
     * Helper function to fill_cluster_embs_in_cache()
     * Retrieve the embeddings of a single object from the KV store in Cascade. 
     * This retrive doesn't involve copying the data.
     * @param retrieved_num_embs the number of embeddings in the cluster
     * @param cluster_emb_key the key of the object to retrieve
     * @param typed_ctxt the context to get the service client reference
     * @param version the version of the object to retrieve
     * @param stable whether to get the stable version of the object
     ***/
     float* single_emb_object_retrieve(int& retrieved_num_embs,
                                             std::string& cluster_emb_key,
                                             DefaultCascadeContextType* typed_ctxt,
                                             persistent::version_t version,
                                             bool stable = 1);

     /*** 
     * Helper function to fill_cluster_embs_in_cache()
     * Retrieve the embeddings of multiple objects from the KV store in Cascade. 
     * This involve copying the data from received blobs.
     ***/
     float* multi_emb_object_retrieve(int& retrieved_num_embs,
                                        std::priority_queue<std::string, std::vector<std::string>, CompareObjKey>& emb_obj_keys,
                                        DefaultCascadeContextType* typed_ctxt,
                                        persistent::version_t version,
                                        bool stable = 1);

     /***
     * Fill in the embeddings of that cluster by getting the clusters' embeddings from KV store in Cascade
     * This function is called when this UDL is first triggered by caller to operator(),
     * it sacrifices the first request to this node, but the following requests will benefit from this cache.
     * In static RAG setting, this function should be called only once at the begining
     * In dynamic RAG setting, this function could be extended to call periodically or upon notification 
     * (The reason of not filling it at initialization, is that initialization is called upon server starts, 
     *  but the data have not been put to the servers yet, this needs to be called after the clusters' embeddings data are put)
     * @param embs_prefix the prefix of embeddings objects that belong to this grouped_embeddings
     * @param typed_ctxt the context to get the service client reference
     * @return 0 on success, -1 on failure
     * @note we load the stabled version of the cluster embeddings
     ***/
     int retrieve_grouped_embeddings(std::string embs_prefix,
                                        DefaultCascadeContextType* typed_ctxt);

     int get_num_embeddings(); 

     int initialize_groupped_embeddings_for_search(int cluster_id = 0);

     /***
      * Search the top K embeddings that are close to one query
      * @param nq: number of queries
      * @param xq: flaten queries to search
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
      */
     void search(int nq, const float* xq, int top_k, float* D, long* I);

     void initialize_cpu_hnsw_search(int cluster_id);

     int hnsw_cpu_search(int nq, const float* xq, int top_k, float* D, long* I);

     /*** 
      * Initialize the CPU flat search index based on the embeddings.
      * initalize it if use faiss_cpu_flat_search()
     ***/
     void initialize_cpu_flat_search();

     /***
      * FAISS knn flat search on CPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/1-Flat.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_cpu_flat_search(int nq, const float* xq, int top_k, float* D, long* I);

     /*** 
      * Initialize the GPU flat search index based on the embeddings.
      * initalize it if use faiss_gpu_flat_search()
     ***/
     void initialize_gpu_flat_search();

     /***
      * FAISS knn flat l2 search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_flat_search(int nq, const float* xq, int top_k, float* D, long* I);

     /*** 
      * Initialize the GPU ivf search index based on the embeddings.
      * initalize it if use faiss_gpu_ivf_flat_search()
     ***/
     void initialize_gpu_ivf_flat_search();

     /***
      * FAISS knn search based on ivf search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_ivf_flat_search(int nq, const float* xq, int top_k, float* D, long* I);

     /***
      * Reset the GroupedEmbeddingsForSearch object
      * This function is called when the UDL is released
      */
     void reset();

     ~GroupedEmbeddingsForSearch() {
          // free(embeddings);
          if (this->embeddings != nullptr) {
               free(this->embeddings);
               this->embeddings = nullptr; 
          }
     }

};

} // namespace cascade
} // namespace derecho
