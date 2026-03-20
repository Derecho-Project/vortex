#include "centroids_search_udl.hpp"


namespace derecho{
namespace cascade{

// CentroidSearchThread

CentroidsSearchOCDPO::CentroidSearchThread::CentroidSearchThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl) : my_thread_id(thread_id), parent(parent_udl) {}

void CentroidsSearchOCDPO::CentroidSearchThread::start(){
    running = true;
    real_thread = std::thread(&CentroidSearchThread::main_loop, this);
}

void CentroidsSearchOCDPO::CentroidSearchThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void CentroidsSearchOCDPO::CentroidSearchThread::signal_stop() {
    std::lock_guard<std::mutex> lock(query_queue_mutex); 
    running = false;
    query_queue_cv.notify_all();
}

void CentroidsSearchOCDPO::CentroidSearchThread::push(std::shared_ptr<EmbeddingQuery> query) {
    std::unique_lock<std::mutex> lock(query_queue_mutex);
    query_queue.push(query);
    query_queue_cv.notify_all();
}

void CentroidsSearchOCDPO::CentroidSearchThread::main_loop() {
    std::unique_lock<std::mutex> lock(query_queue_mutex,std::defer_lock);
    while (running) {
        lock.lock();
        if(query_queue.empty()){
            query_queue_cv.wait(lock);
        }

        if(!running) break;

        auto query = query_queue.front();
        query_queue.pop();
        lock.unlock();

        // search centroids
        std::unique_ptr<std::vector<uint64_t>> clusters = centroid_search(query);

        // push query to the batching thread on the correct clusters
        parent->batch_thread->push(query,std::move(clusters)); 
    }
}

std::unique_ptr<std::vector<uint64_t>> CentroidsSearchOCDPO::CentroidSearchThread::centroid_search(std::shared_ptr<EmbeddingQuery> &query){
    // TODO timestamp logging in this method should be revisited

    // get query embeddings
    TimestampLogger::log(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_START,query->get_node(),query->get_id(),parent->my_id);
    const float *data = query->get_embeddings_pointer();
    TimestampLogger::log(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_END,query->get_node(),query->get_id(),parent->my_id);

    // search the top_num_centroids that are closer to the query
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START,query->get_node(),query->get_id(),parent->my_id);
    long* I = new long[parent->top_num_centroids];
    float* D = new float[parent->top_num_centroids];
    try{
        parent->centroids_embs->search(1, data, parent->top_num_centroids, D, I);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to search the top_num_centroids for the queries." << std::endl;
        dbg_default_error("{}, Failed to search the top_num_centroids for the queries.", __func__);
        return {};
    }
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END,query->get_node(),query->get_id(),parent->my_id);
    dbg_default_trace("[Centroids search ocdpo]: FINISHED knn search for id: {}", query->get_id()); 
  
    // return result
    std::unique_ptr<std::vector<uint64_t>> clusters = std::make_unique<std::vector<uint64_t>>();
    clusters->reserve(parent->top_num_centroids);
    for(uint64_t i=0;i<parent->top_num_centroids;i++){
        clusters->push_back(static_cast<uint64_t>(I[i]));
    }
    
    delete[] I;
    delete[] D;
    return clusters;
}

// BatchingThread

CentroidsSearchOCDPO::BatchingThread::BatchingThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}


void CentroidsSearchOCDPO::BatchingThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&BatchingThread::main_loop, this, typed_ctxt);
}

void CentroidsSearchOCDPO::BatchingThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void CentroidsSearchOCDPO::BatchingThread::signal_stop() {
    std::lock_guard<std::mutex> lock(cluster_queue_mutex); 
    running = false;
    cluster_queue_cv.notify_all();
}

void CentroidsSearchOCDPO::BatchingThread::push(std::shared_ptr<EmbeddingQuery> query,std::unique_ptr<std::vector<uint64_t>> clusters){
    std::unique_lock<std::mutex> lock(cluster_queue_mutex);
        
    for(auto cluster_id : *clusters){
        if(cluster_queue.count(cluster_id) == 0){
            cluster_queue[cluster_id] = std::make_unique<std::vector<std::shared_ptr<EmbeddingQuery>>>();
            cluster_queue[cluster_id]->reserve(parent->max_batch_size);
        }

        cluster_queue[cluster_id]->push_back(query);
    }

    cluster_queue_cv.notify_all();
}

void CentroidsSearchOCDPO::BatchingThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    // TODO timestamp logging in this method should be revisited
    
    std::unique_lock<std::mutex> lock(cluster_queue_mutex, std::defer_lock);
    std::unordered_map<uint64_t,std::chrono::steady_clock::time_point> wait_time;
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    while (running) {
        lock.lock();
        bool empty = true;
        for(auto& item : cluster_queue){
            if(!(item.second->empty())){
                empty = false;
                break;
            }
        }

        if(empty){
            cluster_queue_cv.wait_for(lock,batch_time);
        }
        
        if (!running) break;

        // move queue pointers out of the map and replace with empty vectors
        std::unordered_map<uint64_t,std::unique_ptr<std::vector<std::shared_ptr<EmbeddingQuery>>>> to_send;
        auto now = std::chrono::steady_clock::now();
        for(auto& item : cluster_queue){
            if(wait_time.count(item.first) == 0){
                wait_time[item.first] = now;
            }

            if((item.second->size() >= parent->min_batch_size) || ((now-wait_time[item.first]) >= batch_time)){
                to_send[item.first] = std::move(item.second);
                item.second = std::make_unique<std::vector<std::shared_ptr<EmbeddingQuery>>>();
                item.second->reserve(parent->max_batch_size);
            }
        }
        
        lock.unlock();

        auto num_shards = typed_ctxt->get_service_client_ref().get_number_of_shards<VolatileCascadeStoreWithStringKey>(NEXT_UDL_SUBGROUP_ID);

        // serialize and send batches
        for(auto& item : to_send){
            uint64_t num_sent = 0;
            uint64_t total = item.second->size();
            // send in batches of maximum max_batch_size queries
            while(num_sent < total){
                uint64_t left = total - num_sent;
                uint64_t batch_size = std::min(parent->max_batch_size,left);

                EmbeddingQueryBatcher batcher(parent->emb_dim,batch_size);
                for(uint64_t i=num_sent;i<(num_sent+batch_size);i++){
                    batcher.add_query(item.second->at(i));
                }
                batcher.serialize();

                ObjectWithStringKey obj;
                obj.key = parent->emit_key_prefix + "/cluster" + std::to_string(item.first);
                obj.blob = std::move(*batcher.get_blob());

                typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(obj, NEXT_UDL_SUBGROUP_ID, static_cast<uint32_t>(item.first) % num_shards, true);

                num_sent += batch_size;
            }
        }
        
        // TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END,task_ptr->client_id,task_ptr->query_batch_id,parent->my_id); // TODO this should go somewhere else
        // TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START,task_ptr->client_id,task_ptr->query_batch_id,pair.first); // TODO go somewhere else
        // TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END,task_ptr->client_id,task_ptr->query_batch_id,pair.first); // TODO go somewhere else
        // TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_END,query->get_node(),query->get_id(),parent->my_id); // TODO go somewhere else
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

    // check if local cache contains the centroids' embeddings
    if(cached_centroids_embs == false) {
        if(!retrieve_and_cache_centroids_index(typed_ctxt)){
            return;
        }
    }

    // TODO temporary, to maintain compatibility with the current logging
    uint32_t client_id;
    uint64_t batch_id;
    std::tie(client_id,batch_id) = parse_client_and_batch_id(key_string);

    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_START,client_id,batch_id,this->my_id);

    // create the manager for this batch: this will copy the buffer from the object blob and deserialize the index, so we can create the individual queries wrappers
    std::unique_ptr<EmbeddingQueryBatchManager> batch_manager = std::make_unique<EmbeddingQueryBatchManager>(object.blob.bytes,object.blob.size,emb_dim);
    
    // send queries to worker threads
    for(auto& query : batch_manager->get_queries()){
        search_threads[next_search_thread]->push(query);
        next_search_thread = (next_search_thread + 1) % num_search_threads;
    }
        
    TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_END,client_id,batch_id,this->my_id);
    dbg_default_trace("[Centroids search ocdpo]: FINISHED knn search for key: {}", key_string);
}

void CentroidsSearchOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("centroids_emb_prefix")) this->centroids_emb_prefix = config["centroids_emb_prefix"].get<std::string>();
        if (config.contains("emb_dim")) this->emb_dim = config["emb_dim"].get<int>();
        if (config.contains("top_num_centroids")) this->top_num_centroids = config["top_num_centroids"].get<int>();
        if (config.contains("faiss_search_type")) this->faiss_search_type = config["faiss_search_type"].get<int>();
        if (config.contains("num_search_threads")) this->num_search_threads = config["num_search_threads"].get<int>();
        if (config.contains("min_batch_size")) this->min_batch_size = config["min_batch_size"].get<int>();
        if (config.contains("max_batch_size")) this->max_batch_size = config["max_batch_size"].get<int>();
        if (config.contains("batch_time_us")) this->batch_time_us = config["batch_time_us"].get<int>();
        if (config.contains("emit_key_prefix")) this->emit_key_prefix = config["emit_key_prefix"].get<std::string>();
        if (this->emit_key_prefix.empty() || this->emit_key_prefix.back() != '/') this->emit_key_prefix += '/';
        this->centroids_embs = std::make_unique<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
    }

    // start search threads
    for(uint64_t thread_id = 0; thread_id < this->num_search_threads; thread_id++) {
        search_threads.emplace_back(new CentroidSearchThread(thread_id,this));
    }
    for(auto& search_thread : search_threads) {
        search_thread->start();
    }

    // start batching thread
    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);
}

void CentroidsSearchOCDPO::shutdown() {
    for(auto& search_thread : search_threads) {
        if(search_thread) {
            search_thread->signal_stop();
            search_thread->join();
        }
    }

    if (batch_thread) {
        batch_thread->signal_stop();
        batch_thread->join();
    }
}

} // namespace cascade
} // namespace derecho

