#include "centroids_search_udl.hpp"


namespace derecho{
namespace cascade{



CentroidsSearchOCDPO::ProcessBatchedTasksThread::ProcessBatchedTasksThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}


void CentroidsSearchOCDPO::ProcessBatchedTasksThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&ProcessBatchedTasksThread::main_loop, this, typed_ctxt);
}

void CentroidsSearchOCDPO::ProcessBatchedTasksThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void CentroidsSearchOCDPO::ProcessBatchedTasksThread::signal_stop() {
    running = false;
    thread_signal.notify_all(); 
}

bool CentroidsSearchOCDPO::ProcessBatchedTasksThread::get_queries_and_emebddings(Blob* blob, 
                                                        float*& data, 
                                                        uint32_t& nq, 
                                                        std::vector<std::string>& query_list,
                                                        const uint32_t& client_id,
                                                        const uint32_t& query_batch_id) {
    // If not include_encoder, could directly deserialize the queries and embeddings from the object
    if (!parent->include_encoder){
        try{
            deserialize_embeddings_and_quries_from_bytes(blob->bytes,blob->size,nq,parent->emb_dim,data,query_list);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
            return false;
        }
    } else {
        // compute the embeddings from the query texts
        try{
            // get the query texts from the object
            nlohmann::json query_texts_json = nlohmann::json::parse(reinterpret_cast<const char*>(blob->bytes), 
                                                                    reinterpret_cast<const char*>(blob->bytes) + blob->size);
            query_list = query_texts_json.get<std::vector<std::string>>();
            nq = query_list.size();
            data = new float[parent->emb_dim * nq];
            // compute the embeddings via open AI API call
            if (!api_utils::get_batch_embeddings(query_list, parent->encoder_name, parent->openai_api_key, parent->emb_dim, data)) {
                std::cerr << "Error: failed to get the embeddings from the query texts via OpenAI API." << std::endl;
                dbg_default_error("{}, Failed to get the embeddings from the query texts via OpenAI API.", __func__);
                delete[] data; 
                data = nullptr;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to compute the embeddings from the query texts." << std::endl;
            dbg_default_error("{}, Failed to compute the embeddings from the query texts.", __func__);
            return false;
        }
    }
    return true;
}

void CentroidsSearchOCDPO::ProcessBatchedTasksThread::combine_common_clusters(const long* I, const int nq, std::map<long, std::vector<int>>& cluster_ids_to_query_ids){
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < parent->top_num_centroids; j++) {
            long cluster_id = I[i * parent->top_num_centroids + j];
            if (cluster_ids_to_query_ids.find(cluster_id) == cluster_ids_to_query_ids.end()) {
                cluster_ids_to_query_ids[cluster_id] = std::vector<int>();
            }
            cluster_ids_to_query_ids[cluster_id].push_back(i);
        }
    }
}


void CentroidsSearchOCDPO::ProcessBatchedTasksThread::process_task(std::unique_ptr<batchedTask> task_ptr, DefaultCascadeContextType* typed_ctxt) {
    // 0. check if local cache contains the centroids' embeddings
    if (parent->cached_centroids_embs == false ) {
        if (!parent->retrieve_and_cache_centroids_index(typed_ctxt)) 
            return;
    }
    // 1. get the query embeddings from the object
    TimestampLogger::log(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_START,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);
    float* data;
    uint32_t nq;
    std::vector<std::string> query_list;
    if (!this->get_queries_and_emebddings(&task_ptr->blob, data, nq, query_list, task_ptr->client_id, task_ptr->query_batch_id)) {
        dbg_default_error("Failed to get the query, embeddings from the object, at centroids_search_udl.");
        return;
    }
    TimestampLogger::log(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_END,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);

    // 2. search the top_num_centroids that are close to the query
    long* I = new long[parent->top_num_centroids * nq];
    float* D = new float[parent->top_num_centroids * nq];
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);
    try{
        parent->centroids_embs->search(nq, data, parent->top_num_centroids, D, I);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to search the top_num_centroids for the queries." << std::endl;
        dbg_default_error("{}, Failed to search the top_num_centroids for the queries.", __func__);
        return;
    }

    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);
    /*** 3. emit the result to the subsequent UDL
            trigger the subsequent UDL by evict the queries to shards that contains its top cluster_embs 
            according to affinity set sharding policy
    ***/
    std::map<long, std::vector<int>> cluster_ids_to_query_ids = std::map<long, std::vector<int>>();
    this->combine_common_clusters(I, nq, cluster_ids_to_query_ids);
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);
    for (const auto& pair : cluster_ids_to_query_ids) {
        if (pair.first == -1) {
            dbg_default_error( "Error: [CentroidsSearchOCDPO] for key: {} a selected cluster among top {}, has cluster_id -1", task_ptr->key, parent->top_num_centroids);
            continue;
        }
        std::string new_key = parent->emit_key_prefix + task_ptr->key + "_cluster" + std::to_string(pair.first);
        std::vector<int> query_ids = pair.second;

        // create an bytes object by concatenating: num_queries + float array of emebddings + list of query_text
        uint32_t num_queries = static_cast<uint32_t>(query_ids.size());
        std::string nq_bytes(4, '\0');
        nq_bytes[0] = (num_queries >> 24) & 0xFF;
        nq_bytes[1] = (num_queries >> 16) & 0xFF;
        nq_bytes[2] = (num_queries >> 8) & 0xFF;
        nq_bytes[3] = num_queries & 0xFF;
        float* query_embeddings = new float[parent->emb_dim * num_queries];
        for (uint32_t i = 0; i < num_queries; i++) {
            int query_id = query_ids[i];
            for (int j = 0; j < parent->emb_dim; j++) {
                query_embeddings[i * parent->emb_dim + j] = data[query_id * parent->emb_dim + j];
            }
        }
        std::vector<std::string> query_texts;
        for (uint32_t i = 0; i < num_queries; i++) {
            query_texts.push_back(query_list[query_ids[i]]);
        }
        // serialize the query embeddings and query texts, formated as num_queries + query_embeddings + query_texts
        std::string query_emb_string = nq_bytes +
                                    std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * parent->emb_dim * num_queries) +
                                    nlohmann::json(query_texts).dump();
        ObjectWithStringKey obj;
        obj.key = new_key;
        obj.blob = Blob(reinterpret_cast<const uint8_t*>(query_emb_string.c_str()), query_emb_string.size());
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START,task_ptr->client_id,task_ptr->query_batch_id,pair.first);

        typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(obj, NEXT_UDL_SUBGROUP_ID, static_cast<uint32_t>(pair.first), true); // TODO: change this hard-coded subgroup_id
        std::cout << "I'm here: " << obj.key << " " << NEXT_UDL_SUBGROUP_ID << " shard: " << pair.first << std::endl;
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END,task_ptr->client_id,task_ptr->query_batch_id,pair.first);
        dbg_default_trace("[Centroids search ocdpo]: Emitted key: {}",new_key);
    }
    delete[] I;
    delete[] D;
    if (parent->include_encoder && data) 
        delete[] data;
    // after using the batched task's blob data, release the memory
    task_ptr->blob.memory_mode = derecho::cascade::object_memory_mode_t::DEFAULT;
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_END,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id);
    dbg_default_trace("[Centroids search ocdpo]: FINISHED knn search for key: {}", task_ptr->key);
}


void CentroidsSearchOCDPO::ProcessBatchedTasksThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    std::unique_lock<std::mutex> lock(parent->active_tasks_mutex, std::defer_lock);
    while (running) {
        lock.lock();
        parent->active_tasks_cv.wait(lock, [&] { 
            return parent->new_request || !parent->active_tasks_queue.empty() || !running; 
        });
        if (!running)
            break;
        std::unique_ptr<batchedTask> task = std::move(parent->active_tasks_queue.front());
        parent->active_tasks_queue.pop();
        lock.unlock();
        this->process_task(std::move(task), typed_ctxt);
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool CentroidsSearchOCDPO::retrieve_and_cache_centroids_index(DefaultCascadeContextType* typed_ctxt){
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_LOADING_START,this->my_id,0,0);
    //  Fill centroids embs and keep it in memory cache
    int filled_centroid_embs = this->centroids_embs->retrieve_grouped_embeddings(this->centroids_emb_prefix,typed_ctxt);
    if (filled_centroid_embs == -1) {
        dbg_default_error("Failed to fill the centroids embeddings in cache, at centroids_search_udl.");
        return false;
    }
    int initialized = this->centroids_embs->initialize_groupped_embeddings_for_search();
    if (initialized == -1) {
        dbg_default_error("Failed to initialize the faiss index for the centroids embeddings, at centroids_search_udl.");
        return false;
    }
    cached_centroids_embs = true;
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_LOADING_END,this->my_id,0,0);
    return true;
}


/*** TODO: right now , it incurs a copy of object to the vector, edit to emplace */
void CentroidsSearchOCDPO::ocdpo_handler(const node_id_t sender,
                            const std::string& object_pool_pathname,
                            const std::string& key_string,
                            const ObjectWithStringKey& object,
                            const emit_func_t& emit,
                            DefaultCascadeContextType* typed_ctxt,
                            uint32_t worker_id) {
    /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/centroids_search instead of /rag/emb, i.e. the objp name***/
    dbg_default_trace("[Centroids search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
    // Logging purpose for performance evaluation
    if (key_string == "flush_logs") {
        std::string log_file_name = "node" + std::to_string(my_id) + "_udls_timestamp.dat";
        TimestampLogger::flush(log_file_name);
        std::cout << "Flushed logs to " << log_file_name <<"."<< std::endl;
        return;
    }

    int client_id = -1;
    int query_batch_id = -1;
    bool usable_logging_key = parse_batch_id(key_string, client_id, query_batch_id); // Logging purpose
    if (!usable_logging_key)
        dbg_default_error("Failed to parse client_id and query_batch_id from key: {}, unable to track correctly.", key_string);
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_START,client_id,query_batch_id,this->my_id);

    Blob blob = std::move(const_cast<Blob&>(object.blob));
    blob.memory_mode = derecho::cascade::object_memory_mode_t::EMPLACED;
    // Append the batched queries task to the queue
    new_request = true;
    std::unique_lock<std::mutex> lock(active_tasks_mutex); 

    active_tasks_queue.push(std::make_unique<batchedTask>(key_string, client_id, query_batch_id, std::move(blob)));
    new_request = false;
    lock.unlock();
    active_tasks_cv.notify_one();
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_END,client_id,query_batch_id,this->my_id);
    dbg_default_trace("[Centroids search ocdpo]: FINISHED knn search for key: {}", key_string);
}



void CentroidsSearchOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("centroids_emb_prefix")) this->centroids_emb_prefix = config["centroids_emb_prefix"].get<std::string>();
        if (config.contains("emb_dim")) this->emb_dim = config["emb_dim"].get<int>();
        if (config.contains("top_num_centroids")) this->top_num_centroids = config["top_num_centroids"].get<int>();
        if (config.contains("faiss_search_type")) this->faiss_search_type = config["faiss_search_type"].get<int>();
        if (config.contains("include_encoder")) this->include_encoder = config["include_encoder"].get<bool>();
        if (config.contains("encoder_name")) this->encoder_name = config["encoder_name"].get<std::string>();
        if (config.contains("openai_api_key")) this->openai_api_key = config["openai_api_key"].get<std::string>();
        if (config.contains("emit_key_prefix")) this->emit_key_prefix = config["emit_key_prefix"].get<std::string>();
        if (this->emit_key_prefix.empty() || this->emit_key_prefix.back() != '/') this->emit_key_prefix += '/';
        this->centroids_embs = std::make_unique<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
    }
    this->process_batched_tasks_thread = std::make_unique<ProcessBatchedTasksThread>(this->my_id, this);
    this->process_batched_tasks_thread->start(typed_ctxt);
}



} // namespace cascade
} // namespace derecho
