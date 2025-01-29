

#include "grouped_embeddings_for_search.hpp"

namespace derecho{
namespace cascade{

inline float* aligned_alloc(size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, CACHE_LINE_SIZE, size) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}

PendingEmbeddingQueryBatch::PendingEmbeddingQueryBatch(uint64_t emb_dim,uint64_t max_size){
    this->emb_dim = emb_dim;
    queries.reserve(max_size);
    embeddings = aligned_alloc(max_size * emb_dim * sizeof(float));
    max_queries = max_size;
    num_queries = 0;
}

PendingEmbeddingQueryBatch::~PendingEmbeddingQueryBatch() {
    delete[] embeddings;
}

const float * PendingEmbeddingQueryBatch::get_embeddings(){
    return embeddings;
}

const std::vector<std::shared_ptr<EmbeddingQuery>>& PendingEmbeddingQueryBatch::get_queries(){
    return queries;
}

uint64_t PendingEmbeddingQueryBatch::capacity(){
    return max_queries;
}

uint64_t PendingEmbeddingQueryBatch::size(){
    return num_queries;
}

uint64_t PendingEmbeddingQueryBatch::space_left(){
    return max_queries - num_queries;
}

bool PendingEmbeddingQueryBatch::empty(){
    return num_queries == 0;
}

void PendingEmbeddingQueryBatch::reset(){
    queries.clear();
    num_queries = 0;
}

uint64_t PendingEmbeddingQueryBatch::add_queries(const std::vector<std::shared_ptr<EmbeddingQuery>>& queries_to_add,
        uint64_t query_start_index,
        uint64_t num_to_add,
        const uint8_t *buffer,
        uint32_t embeddings_position,
        uint32_t embeddings_size){
    if(num_to_add > space_left()){
        std::cerr << "Trying to add more queries than the capacity of the buffer" << std::endl;
        return 0;
    }

    // copy embeddings buffer
    uint64_t start_pos = num_queries * emb_dim * sizeof(float);
    std::memcpy(reinterpret_cast<uint8_t*>(embeddings)+start_pos,buffer+embeddings_position,embeddings_size);
    
    // copy queries
    for(uint64_t i=0;i<num_to_add;i++){ 
        uint64_t idx = query_start_index + i;
        queries.push_back(queries_to_add[idx]);
        num_queries++;
    }

    return num_to_add;
}











/*
void queryQueue::resize_queue(size_t new_query_capacity) {
    size_t new_capacity_in_elements = new_query_capacity * emb_dim;
    float* new_embs = aligned_alloc(new_capacity_in_elements * sizeof(float));
    size_t current_size_in_elements = added_query_offset.load();
    memcpy(new_embs, query_embs, current_size_in_elements * sizeof(float));
    free(query_embs); // Free the old memory
    query_embs = new_embs;
    query_embs_capacity = new_query_capacity;
    query_list.reserve(new_query_capacity);
    query_keys.reserve(new_query_capacity);
}

bool queryQueue::add_queries(std::vector<std::string>&& queries, const std::string& key, float* embs, int emb_dim, int num_queries) {
    if (this->emb_dim != emb_dim) {
        return false;
    }
    size_t required_queries = added_query_offset / this->emb_dim + num_queries;
    if (required_queries > query_embs_capacity) {
        size_t new_query_capacity = query_embs_capacity;
        while (required_queries > new_query_capacity) {
            new_query_capacity *= 2;
        }
        resize_queue(new_query_capacity);
    }
    query_list.insert(query_list.end(), std::make_move_iterator(queries.begin()), std::make_move_iterator(queries.end()));
    query_keys.insert(query_keys.end(), num_queries, key);
    // TODO: could do better by only one copy from the blob object at deserialization
    memcpy(query_embs + added_query_offset, embs, num_queries * emb_dim * sizeof(float));
    added_query_offset += num_queries * emb_dim;
    return true;
}

int queryQueue::count_queries() {
    return query_list.size();
}

void queryQueue::reset() {
    query_list.clear();
    query_keys.clear();
    added_query_offset = 0;
}
*/

float* GroupedEmbeddingsForSearch::single_emb_object_retrieve(int& retrieved_num_embs,
                                        std::string& cluster_emb_key,
                                        DefaultCascadeContextType* typed_ctxt,
                                        persistent::version_t version,
                                        bool stable){
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


float* GroupedEmbeddingsForSearch::multi_emb_object_retrieve(int& retrieved_num_embs,
                                   std::priority_queue<std::string, std::vector<std::string>, CompareObjKey>& emb_obj_keys,
                                   DefaultCascadeContextType* typed_ctxt,
                                   persistent::version_t version,
                                   bool stable){
     float* data;
     size_t num_obj = emb_obj_keys.size();
     size_t data_size = 0;
     Blob blobs[num_obj];
     size_t i = 0;
     while (!emb_obj_keys.empty()){
          std::string emb_obj_key = emb_obj_keys.top();
          emb_obj_keys.pop();
          auto get_query_results = typed_ctxt->get_service_client_ref().get(emb_obj_key,version, stable);
          auto& reply = get_query_results.get().begin()->second.get();
          blobs[i] = std::move(const_cast<Blob&>(reply.blob));
          data_size += blobs[i].size / sizeof(float);  
          i++;
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

int GroupedEmbeddingsForSearch::retrieve_grouped_embeddings(std::string embs_prefix,
                                   DefaultCascadeContextType* typed_ctxt){
     bool stable = 1; 
     persistent::version_t version = CURRENT_VERSION;
     // 0. check the keys for this grouped embedding objects stored in cascade
     //    because of the message size, one cluster might need multiple objects to store its embeddings
     auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, embs_prefix);
     std::vector<std::string> listed_emb_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
     if (listed_emb_obj_keys.empty()) {
          std::cerr << "Error: prefix [" << embs_prefix <<"] has no embedding object found in the KV store" << std::endl;
          dbg_default_error("[{}]at {}, Failed to find object prefix {} in the KV store.", gettid(), __func__, embs_prefix);
          return -1;
     }
     std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> emb_obj_keys = filter_exact_matched_keys(listed_emb_obj_keys, embs_prefix);

     // 1. Get the cluster embeddings from KV store in Cascade
     float* data;
     int num_retrieved_embs = 0;
     if (emb_obj_keys.size() == 1) {
          std::string emb_obj_key = emb_obj_keys.top();
          data = single_emb_object_retrieve(num_retrieved_embs, emb_obj_key, typed_ctxt, version, stable);
     } else {
          data = multi_emb_object_retrieve(num_retrieved_embs, emb_obj_keys, typed_ctxt, version ,stable);
     }
     if (num_retrieved_embs == 0) {
          std::cerr << "Error: embs_prefix:" << embs_prefix <<" has no embeddings found in the KV store" << std::endl;
          dbg_default_error("[{}]at {}, There is no embeddings for prefix{} in the KV store.", gettid(), __func__, embs_prefix);
          return -1;
     }
     dbg_default_trace("[{}]: embs_prefix={}, num_emb_objects={} retrieved.", __func__, embs_prefix, num_retrieved_embs);

     // 2. assign the retrieved embeddings
     this->num_embs = num_retrieved_embs;
     this->embeddings = data;
     // this->centroids_embs[cluster_id]= std::make_unique<GroupedEmbeddingsForSearch>(this->emb_dim, retrieved_num_embs, data);
     // int init_search_res = this->initialize_groupped_embeddings_for_search();
     return 0;
}

int GroupedEmbeddingsForSearch::get_num_embeddings(){
     return this->num_embs;
}   

int GroupedEmbeddingsForSearch::initialize_groupped_embeddings_for_search(int cluster_id){
     switch (this->search_type) {
     case SearchType::FaissCpuFlatSearch:
          initialize_cpu_flat_search();
          break;
     case SearchType::FaissGpuFlatSearch:
          initialize_gpu_flat_search();
          break;
     case SearchType::FaissGpuIvfSearch:
          initialize_gpu_ivf_flat_search();
          break;
     case SearchType::HnswlibCpuSearch:
          initialize_cpu_hnsw_search(cluster_id);
          break;
     default:
          std::cerr << "Error: faiss_search_type not supported" << std::endl;
          dbg_default_error("Failed to initialize faiss search type, at clusters_search_udl.");
          return -1;
     }
     initialized_index.store(true);
     return 0;
}

void GroupedEmbeddingsForSearch::search(int nq, const float* xq, int top_k, float* D, long* I){
     switch (this->search_type) {
     case SearchType::FaissCpuFlatSearch:
          faiss_cpu_flat_search(nq, xq, top_k, D, I);
          break;
     case SearchType::FaissGpuFlatSearch:
          faiss_gpu_flat_search(nq, xq, top_k, D, I);
          break;
     case SearchType::FaissGpuIvfSearch:
          faiss_gpu_ivf_flat_search(nq, xq, top_k, D, I);
          break;
     case SearchType::HnswlibCpuSearch:
          hnsw_cpu_search(nq, xq, top_k, D, I);
          break;
     default:
          std::cerr << "Error: faiss_search_type not supported" << std::endl;
          dbg_default_error("Failed to search the top K embeddings, at clusters_search_udl.");
          break;
     }
}

void GroupedEmbeddingsForSearch::initialize_cpu_hnsw_search(int cluster_id) {
     const std::string &name = this->hnsw_config.dataset_name;
     const int& m = this->hnsw_config.m;
     const int& ef = this->hnsw_config.ef_construction;
     const int& ef_search = this->hnsw_config.ef_search;

     const fs::path prebuilt_indices = fs::path("hnsw_index") / name;
     const fs::path cluster_file = prebuilt_indices / 
          ("hnsw_m_" + std::to_string(m) + "_ef_" + std::to_string(ef) + "_cluster_" + std::to_string(cluster_id) + ".bin");
     
     if (!name.empty() && fs::exists(cluster_file)) {
          dbg_default_info("Loading prebuilt index: {}", cluster_file.string());
          this->l2_space = std::make_unique<hnswlib::L2Space>(this->emb_dim);
          this->cpu_hnsw_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(l2_space.get(), cluster_file);
     } else {
          this->l2_space = std::make_unique<hnswlib::L2Space>(this->emb_dim);
          this->cpu_hnsw_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(l2_space.get(), this->num_embs, m, ef);

          for(size_t i = 0; i < this->num_embs; i++) {
               this->cpu_hnsw_index->addPoint(this->embeddings + (i * this->emb_dim), i);
          }
     }

     this->cpu_hnsw_index->setEf(ef_search);
}

int GroupedEmbeddingsForSearch::hnsw_cpu_search(int nq, const float* xq, int top_k, float* D, long* I)  {
     for(size_t i = 0; i < nq; i++) {
          const float* query_vector = xq + (i * this->emb_dim);

          auto results = std::move(this->cpu_hnsw_index->searchKnn(query_vector, top_k));

          for(size_t k = 0; k < top_k; k++) {
               auto [distance, idx] = results.top();
               results.pop();

               // populate distance vector which is (n, k)
               *(D + (i * top_k) + k) = distance;

               // populate index vector which is  also (n, k)
               *(I + (i * top_k) + k) = idx;
          }
     }

     return 0;
}

void GroupedEmbeddingsForSearch::initialize_cpu_flat_search(){
     this->cpu_flatl2_index = std::make_unique<faiss::IndexFlatL2>(this->emb_dim); 
     this->cpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
}


int GroupedEmbeddingsForSearch::faiss_cpu_flat_search(int nq, const float* xq, int top_k, float* D, long* I){
     dbg_default_trace("FAISS CPU flat Search in [GroupedEmbeddingsForSearch] class");
     this->cpu_flatl2_index->search(nq, xq, top_k, D, I);
     return 0;
}

void GroupedEmbeddingsForSearch::initialize_gpu_flat_search(){
     this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
     this->gpu_flatl2_index = std::make_unique<faiss::gpu::GpuIndexFlatL2>(this->gpu_res.get(), this->emb_dim);
     this->gpu_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
}


int GroupedEmbeddingsForSearch::faiss_gpu_flat_search(int nq, const float* xq, int top_k, float* D, long* I){
     dbg_default_trace("FAISS GPU flatl2 Search in [GroupedEmbeddingsForSearch] class" );
     this->gpu_flatl2_index->search(nq, xq, top_k, D, I);
     return 0;
}

void GroupedEmbeddingsForSearch::initialize_gpu_ivf_flat_search(){
     int nlist = 100;
     this->gpu_res = std::make_unique<faiss::gpu::StandardGpuResources>();
     this->gpu_ivf_flatl2_index = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(this->gpu_res.get(), this->emb_dim, nlist, faiss::METRIC_L2);
     this->gpu_ivf_flatl2_index->train(this->num_embs, this->embeddings); // train on the embeddings
     this->gpu_ivf_flatl2_index->add(this->num_embs, this->embeddings); // add vectors to the index
}


int GroupedEmbeddingsForSearch::faiss_gpu_ivf_flat_search(int nq, const float* xq, int top_k, float* D, long* I){
     dbg_default_trace("FAISS GPU ivf Search in [GroupedEmbeddingsForSearch] class" );
     this->gpu_ivf_flatl2_index->search(nq, xq, top_k, D, I);
     return 0;
}


void GroupedEmbeddingsForSearch::reset(){
     if (this->embeddings != nullptr) {
          free(this->embeddings);
          this->embeddings = nullptr; 
     }
     // index cleanup 
     if (this->search_type == SearchType::FaissCpuFlatSearch && this->cpu_flatl2_index != nullptr) {
          this->cpu_flatl2_index->reset();
     } 
     else if (this->search_type == SearchType::FaissGpuFlatSearch && this->gpu_flatl2_index != nullptr) {
          cudaError_t sync_err = cudaDeviceSynchronize();
          if (sync_err == cudaSuccess) {
               this->gpu_flatl2_index->reset();  
               this->gpu_res.reset();
               this->gpu_flatl2_index=nullptr;
               this->gpu_res=nullptr;
               cudaDeviceReset();  
          } else {
               std::cerr << "Error during cudaDeviceSynchronize: " << cudaGetErrorString(sync_err) << std::endl;
          }
     } 
     else if (this->search_type == SearchType::FaissGpuIvfSearch && this->gpu_ivf_flatl2_index != nullptr) {
          cudaError_t sync_err = cudaDeviceSynchronize();
          if (sync_err == cudaSuccess) {
               this->gpu_ivf_flatl2_index->reset();  
               this->gpu_res.reset();
               this->gpu_ivf_flatl2_index=nullptr;
               this->gpu_res=nullptr;
               cudaDeviceReset();  
          } else {
               std::cerr << "Error during cudaDeviceSynchronize: " << cudaGetErrorString(sync_err) << std::endl;
          }
     } else if (this->search_type == SearchType::HnswlibCpuSearch && this -> cpu_flatl2_index != nullptr) {
          this->cpu_hnsw_index.reset();
     }
}


} // namespace cascade
} // namespace derecho
