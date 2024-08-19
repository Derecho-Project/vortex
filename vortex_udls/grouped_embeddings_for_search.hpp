#pragma once

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

namespace derecho{
namespace cascade{

class GroupedEmbeddingsForSearch{
// Class to store group of embeddings, which could be the embeddings of a cluster or embeddings of all centroids

     int faiss_search_type; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
     int emb_dim;  //  e.g. 512. The dimension of each embedding
     int num_embs;  //  e.g. 1000. The number of embeddings in the array

     float* embeddings; 

     std::unique_ptr<faiss::IndexFlatL2> cpu_flatl2_index; // FAISS index object. Initialize if use CPU Flat search
     std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_res;  // FAISS GPU resources. Initialize if use GPU search
     std::unique_ptr<faiss::gpu::GpuIndexFlatL2> gpu_flatl2_index; // FAISS index object. Initialize if use GPU Flat search
     std::unique_ptr<faiss::gpu::GpuIndexIVFFlat> gpu_ivf_flatl2_index; // FAISS index object. Initialize if use GPU IVF search

public:

     GroupedEmbeddingsForSearch(int type, int dim) 
          : faiss_search_type(type), emb_dim(dim){}

     GroupedEmbeddingsForSearch(int dim, int num, float* data) 
          : faiss_search_type(0), emb_dim(dim), num_embs(num), embeddings(data) {}

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
                                             bool stable = 1){
          float* data;
          // 1. get the object from KV store
          auto get_query_results = typed_ctxt->get_service_client_ref().get(cluster_emb_key,version, stable);
          auto& reply = get_query_results.get().begin()->second.get();
          Blob blob = std::move(const_cast<Blob&>(reply.blob));
          blob.memory_mode = derecho::cascade::object_memory_mode_t::EMPLACED; // Avoid copy, use bytes from reply.blob, transfer its ownership to GroupedEmbeddingsForSearch.emb_data
          // 2. get the embeddings from the object
          data = const_cast<float*>(reinterpret_cast<const float *>(blob.bytes));
          size_t num_points = blob.size / sizeof(float);
          retrieved_num_embs += num_points / this->emb_dim;
          return data;
     }

     /*** 
     * Helper function to fill_cluster_embs_in_cache()
     * Retrieve the embeddings of multiple objects from the KV store in Cascade. 
     * This involve copying the data from received blobs.
     ***/
     float* multi_emb_object_retrieve(int& retrieved_num_embs,
                                        std::vector<std::string>& emb_obj_keys,
                                        DefaultCascadeContextType* typed_ctxt,
                                        persistent::version_t version,
                                        bool stable = 1){
          float* data;
          size_t num_obj = emb_obj_keys.size();
          size_t data_size = 0;
          Blob blobs[num_obj];
          for (size_t i = 0; i < num_obj; i++) {
               auto get_query_results = typed_ctxt->get_service_client_ref().get(emb_obj_keys[i],version, stable);
               auto& reply = get_query_results.get().begin()->second.get();
               blobs[i] = std::move(const_cast<Blob&>(reply.blob));
               data_size += blobs[i].size / sizeof(float);
          }
          // 2. copy the embeddings from the blobs to the data
          data = (float*)malloc(data_size * sizeof(float));
          size_t offset = 0;
          for (size_t i = 0; i < num_obj; i++) {
               memcpy(data + offset, blobs[i].bytes, blobs[i].size);
               offset += blobs[i].size / sizeof(float);
          }
          retrieved_num_embs = data_size / this->emb_dim;
          return data;
     }

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
                                        DefaultCascadeContextType* typed_ctxt){
          bool stable = 1; 
          persistent::version_t version = CURRENT_VERSION;
          // 0. check the keys for this grouped embedding objects stored in cascade
          //    because of the message size, one cluster might need multiple objects to store its embeddings
          auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, embs_prefix);
          std::vector<std::string> emb_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
          if (emb_obj_keys.empty()) {
               std::cerr << "Error: prefix [" << embs_prefix <<"] has no embedding object found in the KV store" << std::endl;
               dbg_default_error("[{}]at {}, Failed to find object prefix {} in the KV store.", gettid(), __func__, embs_prefix);
               return -1;
          }

          // 1. Get the cluster embeddings from KV store in Cascade
          float* data;
          int num_retrieved_embs = 0;
          if (emb_obj_keys.size() == 1) {
               data = single_emb_object_retrieve(num_retrieved_embs, emb_obj_keys[0], typed_ctxt, version, stable);
          } else {
               data = multi_emb_object_retrieve(num_retrieved_embs, emb_obj_keys, typed_ctxt, version ,stable);
          }
          if (num_retrieved_embs == 0) {
               std::cerr << "Error: embs_prefix:" << embs_prefix <<" has no embeddings found in the KV store" << std::endl;
               dbg_default_error("[{}]at {}, There is no embeddings for prefix{} in the KV store.", gettid(), __func__, embs_prefix);
               return -1;
          }
          dbg_default_trace("[{}]: embs_prefix={}, num_emb_objects={} retrieved.", __func__, embs_prefix, num_retrieved_embs);
          
          // 2. assign the retrieved embeddings to the object
          this->num_embs = num_retrieved_embs;
          this->embeddings = data;
          // this->centroids_embs[cluster_id]= std::make_unique<GroupedEmbeddingsForSearch>(this->emb_dim, retrieved_num_embs, data);
          int init_search_res = this->initialize_groupped_embeddings_for_search();
          return init_search_res;
     }

     int get_num_embeddings(){
          return this->num_embs;
     }   

     int initialize_groupped_embeddings_for_search(){
          if (this->faiss_search_type == 0){
               initialize_cpu_flat_search();
          } else if (this->faiss_search_type == 1){
               initialize_gpu_flat_search();
          } else if (this->faiss_search_type == 2){
               initialize_gpu_ivf_flat_search();
          } else {
               std::cerr << "Error: faiss_search_type not supported" << std::endl;
               dbg_default_error("Failed to initialize faiss search type, at clusters_search_udl.");
               return -1;
          }
          return 0;
     }

     void search(int nq, float* xq, int top_k, float* D, long* I){
          if (this->faiss_search_type == 0){
               faiss_cpu_flat_search(nq, xq, top_k, D, I);
          } else if (this->faiss_search_type == 1){
               faiss_gpu_flat_search(nq, xq, top_k, D, I);
          } else if (this->faiss_search_type == 2){
               faiss_gpu_ivf_flat_search(nq, xq, top_k, D, I);
          } else {
               std::cerr << "Error: faiss_search_type not supported" << std::endl;
               dbg_default_error("Failed to search the top K embeddings, at clusters_search_udl.");
          }
     }

     /*** 
      * Initialize the CPU flat search index based on the embeddings.
      * initalize it if use faiss_cpu_flat_search()
     ***/
     void initialize_cpu_flat_search(){
          this->cpu_flatl2_index = std::make_unique<faiss::IndexFlatL2>(this->emb_dim); 
          this->cpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn flat search on CPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/1-Flat.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_cpu_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          dbg_default_trace("FAISS CPU flat Search in [GroupedEmbeddingsForSearch] class");
          this->cpu_flatl2_index->search(nq, xq, top_k, D, I);
          return 0;
     }

     /*** 
      * Initialize the GPU flat search index based on the embeddings.
      * initalize it if use faiss_gpu_flat_search()
     ***/
     void initialize_gpu_flat_search(){
          this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
          this->gpu_flatl2_index = std::make_unique<faiss::gpu::GpuIndexFlatL2>(this->gpu_res.get(), this->emb_dim);
          this->gpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn flat l2 search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          dbg_default_trace("FAISS GPU flatl2 Search in [GroupedEmbeddingsForSearch] class" );
          int k = top_k;
          this->gpu_flatl2_index->search(nq, xq, k, D, I);
          return 0;
     }

     /*** 
      * Initialize the GPU ivf search index based on the embeddings.
      * initalize it if use faiss_gpu_ivf_flat_search()
     ***/
     void initialize_gpu_ivf_flat_search(){
          int nlist = 100;
          this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
          this->gpu_ivf_flatl2_index = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(this->gpu_res.get(), this->emb_dim, nlist, faiss::METRIC_L2);
          this->gpu_ivf_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
     }

     /***
      * FAISS knn search based on ivf search on GPU
      * https://github.com/facebookresearch/faiss/blob/main/tutorial/cpp/4-GPU.cpp
      * @param nq: number of queries
      * @param xq: flaten queries to search 
      * @param top_k: number of top embeddings to return
      * @param D: distance array to store the distance of the top_k embeddings
      * @param I: index array to store the index of the top_k embeddings
     ***/
     int faiss_gpu_ivf_flat_search(int nq, float* xq, int top_k, float* D, long* I){
          dbg_default_error("FAISS GPU ivf flatl2 Search in [GroupedEmbeddingsForSearch] class");
          this->gpu_ivf_flatl2_index->search(nq, xq, top_k, D, I);
          // print results
          printf("I (5 first results)=\n");
          for (int i = 0; i < 5; i++) {
               for (int j = 0; j < top_k; j++)
                    printf("%5ld ", I[i * top_k + j]);
               printf("\n");
          }

          printf("I (5 last results)=\n");
          for (int i = nq - 5; i < nq; i++) {
               for (int j = 0; j < top_k; j++)
                    printf("%5ld ", I[i * top_k + j]);
               printf("\n");
          }
          return 0;
     }

     ~GroupedEmbeddingsForSearch() {
          // free(embeddings);
          delete[] this->embeddings;
     }

};

} // namespace cascade
} // namespace derecho