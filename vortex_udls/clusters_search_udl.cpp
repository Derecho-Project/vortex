#include <iostream>
#include <thread>
#include <chrono>

// #include "search_worker.hpp"
#include "clusters_search_udl.hpp"


namespace derecho{
namespace cascade{

queryQueue::queryQueue(int emb_dim): emb_dim(emb_dim) {
    query_list.reserve(MAX_NUM_QUERIES_PER_BATCH);
    query_keys.reserve(MAX_NUM_QUERIES_PER_BATCH);
    query_embs = new float[MAX_NUM_QUERIES_PER_BATCH * emb_dim];
    added_query_offset = 0;
}

queryQueue::~queryQueue() {
    delete[] query_embs;
}

bool queryQueue::add_query(std::string&& query, std::string&& key, float* emb, int emb_dim) {
    if (added_query_offset + emb_dim > MAX_NUM_QUERIES_PER_BATCH * emb_dim) {
        return false;
    }
    query_list.emplace_back(std::move(query));
    query_keys.emplace_back(std::move(key));
    memcpy(query_embs + added_query_offset, emb, emb_dim * sizeof(float));
    added_query_offset += emb_dim;
    return true;
}

bool queryQueue::add_batched_queries(std::vector<std::string>&& queries, const std::string& key, float* embs, int emb_dim, int num_queries) {
    if (added_query_offset + num_queries * emb_dim > MAX_NUM_QUERIES_PER_BATCH * emb_dim) {
        return false;
    }
    query_list.insert(query_list.end(), std::make_move_iterator(queries.begin()), std::make_move_iterator(queries.end()));
    query_keys.insert(query_keys.end(), num_queries, key);
    memcpy(query_embs + added_query_offset, embs, num_queries * emb_dim * sizeof(float));
    added_query_offset += num_queries * emb_dim;
    return true;
}

bool queryQueue::could_add_query_nums(uint32_t num_queries) {
    bool has_space = added_query_offset + num_queries * emb_dim <= MAX_NUM_QUERIES_PER_BATCH * emb_dim;
    // This should be rare occurance, as it blocks the main thread. 
    // Increase the MAX_NUM_QUERIES_PER_BATCH if this happens, meaning the local cache is too small for the high query rate
    if (!has_space) {
        dbg_default_error("Failed to add queries to the queue, not enough space. num_queries = {}", num_queries);
    }
    return has_space;
}

int queryQueue::count_queries() {
    return query_list.size();
}

void queryQueue::reset() {
    query_list.clear();
    query_keys.clear();
    added_query_offset = 0;
}

ClustersSearchOCDPO::ClusterSearchWorker::ClusterSearchWorker(uint64_t thread_id, 
                                        ClustersSearchOCDPO* parent_udl)
        : my_thread_id(thread_id), parent(parent_udl), running(false){
        use_shadow_flag.store(0);
        query_buffer = std::make_unique<queryQueue>(parent->emb_dim);
        shadow_query_buffer = std::make_unique<queryQueue>(parent->emb_dim);
    }

void ClustersSearchOCDPO::ClusterSearchWorker::construct_new_keys(std::vector<std::string>& new_keys,
                                                       const std::vector<std::string>& query_keys, 
                                                       const std::vector<std::string>& query_list) {
    for (size_t i = 0; i < query_keys.size(); ++i) {
        const std::string& key_string = query_keys[i];
        const std::string& query = query_list[i];
        std::string hashed_query;
        try {
            /*** TODO: do we need 32 bytes of hashed key? will simply int be sufficient? */
            uint8_t digest[32];
            openssl::Hasher sha256(openssl::DigestAlgorithm::SHA256);
            const char* query_cstr = query.c_str();
            sha256.hash_bytes(query_cstr, strlen(query_cstr), digest);
            std::ostringstream oss;
            for (int i = 0; i < 32; ++i) {
                // Output each byte as a decimal value (0-255) without any leading zeros
                oss << std::dec << static_cast<int>(digest[i]);
            }
            hashed_query = oss.str();
        } catch(openssl::openssl_error& ex) {
            dbg_default_error("Unable to compute SHA256 of typename. string = {}, exception name = {}", query, ex.what());
            throw;
        }
        std::string new_key = key_string + "_qid" + hashed_query;
        new_keys.push_back(new_key);
    }
}


bool ClustersSearchOCDPO::ClusterSearchWorker::check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt){
    uint32_t cluster_id = static_cast<uint32_t>(this->cluster_id);
    // Acquire a unique lock to modify the cluster search index
    std::unique_lock<std::shared_mutex> write_lock(parent->cluster_search_index_mutex);
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START, parent->my_id, cluster_id, 0);
    // Double-check if the cluster was inserted by another thread
    if (parent->cluster_search_index->initialized_index.load()) {
        return true;
    } 
    std::string cluster_prefix = CLUSTER_EMB_OBJECTPOOL_PREFIX + std::to_string(this->cluster_id);
    int filled_cluster_embs = parent->cluster_search_index->retrieve_grouped_embeddings(cluster_prefix, typed_ctxt);
    if (filled_cluster_embs == -1) {
        std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
        dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
        return false;
    }
    int initialized = parent->cluster_search_index->initialize_groupped_embeddings_for_search(this->cluster_id);
    if (initialized == -1) {
        std::cerr << "Error: failed to initialize the index for the cluster embeddings" << std::endl;
        dbg_default_error("Failed to initialize the index for the cluster embeddings, at clusters_search_udl.");
        return false;
    }
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END, parent->my_id, cluster_id, 0);

    return true;
}

/***
* Run ANN algorithm on the query and emit the results
*/
void ClustersSearchOCDPO::ClusterSearchWorker::run_cluster_search_and_emit(DefaultCascadeContextType* typed_ctxt, 
                                                                queryQueue* query_buffer) {
    int nq = query_buffer->count_queries();
    long* I = new long[parent->top_k * nq];
    float* D =  new float[parent->top_k * nq];
    parent->cluster_search_index->search(nq, query_buffer->query_embs, parent->top_k, D, I);
    if (!I || !D) {
        dbg_default_error("Failed to batch search for cluster: {}", this->cluster_id);
        return;
    }
    std::vector<std::string> new_keys;
    this->construct_new_keys(new_keys, query_buffer->query_keys, query_buffer->query_list);
    size_t num_queries = query_buffer->query_list.size();
    for (size_t k = 0; k < num_queries; ++k) {
        ObjectWithStringKey obj;
        obj.key = std::string(EMIT_AGGREGATE_PREFIX) + "/" + new_keys[k];
        std::string query_emit_content = serialize_cluster_search_result(parent->top_k, I, D, k, query_buffer->query_list[k]);
        obj.blob = Blob(reinterpret_cast<const uint8_t*>(query_emit_content.c_str()), query_emit_content.size());
        int client_id = -1, query_batch_id = -1;
        parse_batch_id(obj.key, client_id, query_batch_id);
        TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_EMIT_START,client_id,query_batch_id, this->cluster_id);
        typed_ctxt->get_service_client_ref().put_and_forget(obj);
    }

    delete[] I;
    delete[] D;
}

// check pending queries, flip the shadow_flag if needed
bool ClustersSearchOCDPO::ClusterSearchWorker::enough_pending_queries(int num){
    // flip the shadow buffer if the adding_buffer size exceeds the batch_min_size 
    if (use_shadow_flag.load() && shadow_query_buffer->count_queries() > num) {
        return true;
    } else if (!use_shadow_flag.load() && query_buffer->count_queries() > num) {
        return true;
    }
    return false;
}

void ClustersSearchOCDPO::ClusterSearchWorker::main_loop(DefaultCascadeContextType* typed_ctxt) {
    
    auto wait_start = std::chrono::steady_clock::now();
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    while (running) {
        std::unique_lock<std::mutex> query_buff_lock(query_buffer_mutex);
        // check if there are queries to be processed on the current processing buffer
        query_buffer_cv.wait_for(query_buff_lock, std::chrono::microseconds(parent->batch_time_us),[this]() {
            return this->enough_pending_queries(parent->batch_min_size); ; 
        });
        if (!running) break;
        if ( !this->enough_pending_queries(0)) {
            // if no queries in the buffer, then continue to wait for the next batch
            continue;
        }

        use_shadow_flag.fetch_xor(true);
        query_buff_lock.unlock();
        // notify the push_thread that a new buffer is ready
        query_buffer_cv.notify_one();
        

        if (!running) break;

        if (!parent->cluster_search_index->initialized_index.load()){
            if(!this->check_and_retrieve_cluster_index(typed_ctxt)){
                continue;
            }
        }
        queryQueue* cur_query_buffer = nullptr;
        if (use_shadow_flag.load()) {
            cur_query_buffer = query_buffer.get();
        } else {
            cur_query_buffer = shadow_query_buffer.get();
        }
        run_cluster_search_and_emit(typed_ctxt, cur_query_buffer);
        // reset the current query buffer
        cur_query_buffer->reset();
        wait_start = std::chrono::steady_clock::now();
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::push_to_query_buffer(int cluster_id, 
                                                                    const Blob& blob, 
                                                                    const std::string& key) {
    if (this->cluster_id == -1){
        this->cluster_id = cluster_id;
    } else if (this->cluster_id != cluster_id) {
        std::cerr << "Error: cluster_id mismatched" << std::endl;
        dbg_default_error("Cluster ID mismatched, at clusters_search_udl.");
        return;
    }
    // 1. get the query embeddings from the object
    float* data;
    uint32_t nq;
    std::vector<std::string> query_list;
    try{
        deserialize_embeddings_and_quries_from_bytes(blob.bytes, blob.size, nq, parent->emb_dim, data, query_list);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
        dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
        return;
    }
    // 2. add the queries to the queueing batch
    std::unique_lock<std::mutex> lock(query_buffer_mutex);
    query_buffer_cv.wait(lock, [this, nq] { 
        if (use_shadow_flag.load()) {
            return shadow_query_buffer->could_add_query_nums(nq);
        }
        return query_buffer->could_add_query_nums(nq);
    });
    // this step incurs a copy of the query embedding float[], to make it aligned with the other embeddings in the buffer for batching
    if (use_shadow_flag.load()) {
        shadow_query_buffer->add_batched_queries(std::move(query_list), key, data, parent->emb_dim, nq);
    } else {
        query_buffer->add_batched_queries(std::move(query_list), key, data, parent->emb_dim, nq);
    }
    query_buffer_cv.notify_one();
}

void ClustersSearchOCDPO::ClusterSearchWorker::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&ClusterSearchWorker::main_loop, this, typed_ctxt);
}

void ClustersSearchOCDPO::ClusterSearchWorker::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::signal_stop() {
    running = false;
    query_buffer_cv.notify_all();
}

    
void ClustersSearchOCDPO::ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) {
    /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/clusteres_search instead of /rag/emb, i.e. the objp name***/
    dbg_default_trace("[Clusters search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
    // 0. parse the key, get the cluster ID
    int cluster_id;
    bool extracted_clusterid = parse_number(key_string, CLUSTER_KEY_DELIMITER, cluster_id); 
    if (!extracted_clusterid) {
        std::cerr << "Error: cluster ID not found in the key_string" << std::endl;
        dbg_default_error("Failed to find cluster ID from key: {}, at clusters_search_udl.", key_string);
        return;
    }
    int client_id = -1;
    int query_batch_id = -1;
    bool usable_logging_key = parse_batch_id(key_string, client_id, query_batch_id); // Logging purpose
    if (!usable_logging_key)
        dbg_default_error("Failed to parse client_id and query_batch_id from key: {}, unable to track correctly.", key_string);
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_START,client_id,query_batch_id,cluster_id);
    // 1. Move the object to the active queue to be processed
    this->cluster_search_thread->push_to_query_buffer(cluster_id, object.blob, key_string);
    dbg_default_trace("[Cluster search ocdpo]: PUT {} to active queue.", key_string );
}


void ClustersSearchOCDPO::start_threads(DefaultCascadeContextType* typed_ctxt) {
    /*** TODO: this could be extended to thread-pool  */
    if (!cluster_search_thread) {
        uint64_t thread_id = 0;
        cluster_search_thread = std::make_unique<ClusterSearchWorker>(thread_id, 
                                this);
        cluster_search_thread->start(typed_ctxt);
    }
}


void ClustersSearchOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("emb_dim")) {
            this->emb_dim = config["emb_dim"].get<int>();
        }
        if (config.contains("top_k")) {
            this->top_k = config["top_k"].get<uint32_t>();
        }
        if (config.contains("faiss_search_type")) {
            this->faiss_search_type = config["faiss_search_type"].get<int>();
        }
        if (config.contains("dataset_name")) {
            this->dataset_name = config["dataset_name"].get<std::string>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
    }
    this->cluster_search_index = std::make_shared<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim, this->dataset_name);
    this->start_threads(typed_ctxt);
}

void ClustersSearchOCDPO::shutdown() {
    if (cluster_search_thread) {
        cluster_search_thread->signal_stop();
        cluster_search_thread->join();
    }
    std::unique_lock<std::shared_mutex> lock(cluster_search_index_mutex);
    // Clean up index resources
    cluster_search_index->reset();
}




} // namespace cascade
} // namespace derecho
