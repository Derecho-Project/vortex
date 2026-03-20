#include <iostream>
#include <thread>
#include <chrono>

// #include "search_worker.hpp"
#include "clusters_search_udl.hpp"


namespace derecho{
namespace cascade{

ClustersSearchOCDPO::ClusterSearchWorker::ClusterSearchWorker(uint64_t thread_id, 
                                        ClustersSearchOCDPO* parent_udl)
        : my_thread_id(thread_id), parent(parent_udl), running(false){
    // pre-allocate batch buffers
    pending_batches.reserve(INITIAL_PENDING_BATCHES);
    for(uint64_t i=0;i<INITIAL_PENDING_BATCHES;i++){
        PendingEmbeddingQueryBatch *pending_batch = new PendingEmbeddingQueryBatch(parent->emb_dim,parent->max_process_batch_size);
        pending_batches.emplace_back(pending_batch);
    }
}

void ClustersSearchOCDPO::ClusterSearchWorker::push_queries(uint64_t cluster_id, std::unique_ptr<EmbeddingQueryBatchManager> batch_manager,const uint8_t *buffer){
    const std::vector<std::shared_ptr<EmbeddingQuery>>& queries = batch_manager->get_queries();
    uint64_t num_queries = batch_manager->count();
    uint64_t queries_added = 0;

    std::unique_lock<std::mutex> lock(query_buffer_mutex);

    // find the next free buffer and add the queries there (create a new one if no buffer has space) 
    while(queries_added < num_queries){
        // find the next free buffer
        int64_t free_batch = next_batch;
        uint64_t space_left = pending_batches[free_batch]->space_left();
        while (space_left == 0){
            // cycle to the next
            free_batch = (free_batch + 1) % pending_batches.size();
            if(free_batch == current_batch){ // skip the batch being currently processed
                free_batch = (free_batch + 1) % pending_batches.size();
            }
            
            if(free_batch >= next_batch){
                break;
            }

            space_left = pending_batches[free_batch]->space_left();
        }

        if(space_left == 0){ // could not find a buffer with space, we need to create a new one
            PendingEmbeddingQueryBatch *pending_batch = new PendingEmbeddingQueryBatch(parent->emb_dim,parent->max_process_batch_size);
            pending_batches.emplace_back(pending_batch);
            free_batch = pending_batches.size()-1;
            space_left = pending_batches[free_batch]->space_left();
        }

        // add as many queries as possible
        next_batch = free_batch;
        uint64_t query_start_index = queries_added;
        uint64_t num_to_add = std::min(space_left,num_queries-queries_added);
        uint32_t embeddings_position = batch_manager->get_embeddings_position(query_start_index);
        uint32_t embeddings_size = batch_manager->get_embeddings_size(num_to_add);
        
        pending_batches[next_batch]->add_queries(queries,query_start_index,num_to_add,buffer,embeddings_position,embeddings_size);
        queries_added += num_to_add;

        // if we complete filled the buffer, cycle to the next
        if(pending_batches[next_batch]->space_left() == 0){
            next_batch = (next_batch + 1) % pending_batches.size();
            if(next_batch == current_batch){ // skip the batch being currently processed
                next_batch = (next_batch + 1) % pending_batches.size();
            }
        }
    }
        
    query_buffer_cv.notify_one();
}

void ClustersSearchOCDPO::ClusterSearchWorker::main_loop(DefaultCascadeContextType* typed_ctxt) {
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    std::shared_ptr<PendingEmbeddingQueryBatch> batch;
    while (running) {
        batch.reset(); // clear the pointer

        std::unique_lock<std::mutex> lock(query_buffer_mutex);
        
        current_batch = -1;
        if(pending_batches[next_to_process]->empty()){
            query_buffer_cv.wait_for(lock,batch_time);
        }

        if(!pending_batches[next_to_process]->empty()){
            current_batch = next_to_process;
            next_to_process = (next_to_process + 1) % pending_batches.size();
            batch = pending_batches[current_batch];

            // if we are gonna process now the same batch that is being filled by the main thread, we need to set the next batch to start being filled
            if(current_batch == next_batch){
                next_batch = (next_batch + 1) % pending_batches.size();
            }
        }

        lock.unlock();
        
        if (!running) break;

        if(current_batch < 0){
            continue;
        }

        //  process batch
        std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> results = process_batch(batch);

        // push result to the batching thread
        parent->batch_thread->push_results(std::move(results));

        // clear batch so it can receive queries again
        batch->reset();
    }
}

std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> ClustersSearchOCDPO::ClusterSearchWorker::process_batch(std::shared_ptr<PendingEmbeddingQueryBatch> batch){
    const float * embeddings = batch->get_embeddings();
    const std::vector<std::shared_ptr<EmbeddingQuery>>& queries = batch->get_queries();
    uint64_t num_queries = batch->size();

    // search
    long* I = new long[parent->top_k * num_queries];
    float* D = new float[parent->top_k * num_queries];
    parent->cluster_search_index->search(num_queries, embeddings, parent->top_k, D, I);
    
    if (!I || !D) {
        dbg_default_error("Failed to batch search for cluster: {}", parent->cluster_id);
        delete[] I;
        delete[] D;
        return nullptr;
    }

    // create result vector
    std::shared_ptr<long> ids(I);
    std::shared_ptr<float> dist(D);
    std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> results = std::make_unique<std::vector<std::shared_ptr<ClusterSearchResult>>>();
    for(uint64_t i=0;i<num_queries;i++){
        ClusterSearchResult *res = new ClusterSearchResult(queries[i],ids,dist,i,parent->top_k,parent->cluster_id);
        results->emplace_back(res);
    }

    return results;
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

// BatchingThread

ClustersSearchOCDPO::BatchingThread::BatchingThread(uint64_t thread_id, ClustersSearchOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}


void ClustersSearchOCDPO::BatchingThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&BatchingThread::main_loop, this, typed_ctxt);
}

void ClustersSearchOCDPO::BatchingThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void ClustersSearchOCDPO::BatchingThread::signal_stop() {
    std::lock_guard<std::mutex> lock(shard_queue_mutex);
    running = false;
    shard_queue_cv.notify_all();
}

void ClustersSearchOCDPO::BatchingThread::push_results(std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> results){
    uint32_t num_shards = capi.get_number_of_shards<VolatileCascadeStoreWithStringKey>(AGGREGATE_SUBGROUP_INDEX);
    std::unique_lock<std::mutex> lock(shard_queue_mutex);

    for(auto& result : *results){
        uint32_t shard = result->get_query_id() % num_shards;
        if(shard_queue.count(shard) == 0){
            shard_queue[shard] = std::make_unique<std::vector<std::shared_ptr<ClusterSearchResult>>>();
            shard_queue[shard]->reserve(parent->max_batch_size);
        }

        shard_queue[shard]->push_back(result);
    }

    shard_queue_cv.notify_all();
}

void ClustersSearchOCDPO::BatchingThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    // TODO timestamp logging in this method should be revisited

    std::unique_lock<std::mutex> lock(shard_queue_mutex, std::defer_lock);
    std::unordered_map<uint32_t,std::chrono::steady_clock::time_point> wait_time;
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    while (running) {
        lock.lock();
        bool empty = true;
        for(auto& item : shard_queue){
            if(!(item.second->empty())){
                empty = false;
                break;
            }
        }

        if(empty){
            shard_queue_cv.wait_for(lock,batch_time);
        }

        if (!running) break;

        // move queue pointers out of the map and replace with empty vectors
        std::unordered_map<uint32_t,std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>>> to_send;
        auto now = std::chrono::steady_clock::now();
        for(auto& item : shard_queue){
            if(wait_time.count(item.first) == 0){
                wait_time[item.first] = now;
            }

            if((item.second->size() >= parent->min_batch_size) || ((now-wait_time[item.first]) >= batch_time)){
                to_send[item.first] = std::move(item.second);
                item.second = std::make_unique<std::vector<std::shared_ptr<ClusterSearchResult>>>();
                item.second->reserve(parent->max_batch_size);
            }
        }

        lock.unlock();

        // serialize and send batches
        for(auto& item : to_send){
            uint64_t num_sent = 0;
            uint64_t total = item.second->size();

            // send in batches of maximum max_batch_size queries
            while(num_sent < total){
                uint64_t left = total - num_sent;
                uint64_t batch_size = std::min(static_cast<uint64_t>(parent->max_batch_size),left);

                ClusterSearchResultBatcher batcher(parent->top_k,batch_size);
                for(uint64_t i=num_sent;i<(num_sent+batch_size);i++){
                    batcher.add_result(item.second->at(i));
                }
                batcher.serialize();

                ObjectWithStringKey obj;
                obj.key = EMIT_AGGREGATE_PREFIX "/results_cluster" + std::to_string(parent->cluster_id);
                obj.blob = std::move(*batcher.get_blob());

                typed_ctxt->get_service_client_ref().put_and_forget<VolatileCascadeStoreWithStringKey>(obj, AGGREGATE_SUBGROUP_INDEX, item.first, true);

                num_sent += batch_size;
            }
        }
    }
}

bool ClustersSearchOCDPO::check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt){
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START, my_id, this->cluster_id, 0);
    std::string cluster_prefix = CLUSTER_EMB_OBJECTPOOL_PREFIX + std::to_string(this->cluster_id);
    int filled_cluster_embs = cluster_search_index->retrieve_grouped_embeddings(cluster_prefix, typed_ctxt);
    if (filled_cluster_embs == -1) {
        std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
        dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
        return false;
    }
    int initialized = cluster_search_index->initialize_groupped_embeddings_for_search(this->cluster_id);
    if (initialized == -1) {
        std::cerr << "Error: failed to initialize the index for the cluster embeddings" << std::endl;
        dbg_default_error("Failed to initialize the index for the cluster embeddings, at clusters_search_udl.");
        return false;
    }
    TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END, my_id, this->cluster_id, 0);

    return true;
}
    
void ClustersSearchOCDPO::ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) {
    // TODO timestamp logging need to be revisited

    /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/clusters_search instead of /rag/emb, i.e. the objp name***/
    dbg_default_trace("[Clusters search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
    
    // 0. parse the key, get the cluster ID
    uint64_t parsed_cluster_id = parse_cluster_id(key_string);

    // 1. check if we have the cluster index
    if (this->cluster_id == -1) {
        this->cluster_id = parsed_cluster_id;

        // Check if the cluster search index is initialized
        if (!cluster_search_index->initialized_index.load()){
            if(!check_and_retrieve_cluster_index(typed_ctxt)){
                this->cluster_id = -1;
                return;
            }
        }
    } else if (this->cluster_id != parsed_cluster_id) {
        std::cerr << "Error: cluster ID mismatched" << std::endl;
        dbg_default_error("Cluster ID mismatched, at clusters_search_udl.");
        return;
    }

    //TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_START,my_id,0,parsed_cluster_id); // TODO revise
   
    // 3. batch manager 
    std::unique_ptr<EmbeddingQueryBatchManager> batch_manager = std::make_unique<EmbeddingQueryBatchManager>(object.blob.bytes,object.blob.size,emb_dim,false);
    
    // 4. add queries to the to the appropriate queue in the next thread
    this->cluster_search_threads[next_thread]->push_queries(parsed_cluster_id, std::move(batch_manager),object.blob.bytes);
    next_thread = (next_thread + 1) % this->num_threads; // cycle
    
    dbg_default_trace("[Cluster search ocdpo]: PUT {} to active queue on thread {}.", key_string, next_thread);
}


void ClustersSearchOCDPO::start_threads(DefaultCascadeContextType* typed_ctxt) {
    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);

    for (int thread_id = 0; thread_id < this->num_threads; thread_id++) {
        cluster_search_threads.emplace_back(std::make_unique<ClusterSearchWorker>(static_cast<uint64_t>(thread_id), this));
    }
    for (auto& cluster_search_thread : cluster_search_threads) {
        cluster_search_thread->start(typed_ctxt);
    }
}

void ClustersSearchOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    HNSWConfig hnsw_config;
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
            hnsw_config.dataset_name = config["dataset_name"].get<std::string>();
        }
        if (config.contains("hnsw_m")) {
            hnsw_config.m = config["hnsw_m"].get<int>();
        }
        if (config.contains("hnsw_ef_construction")) {
            hnsw_config.ef_construction = config["hnsw_ef_construction"].get<int>();
        }
        if (config.contains("hnsw_ef_search")) {
            hnsw_config.ef_search = config["hnsw_ef_search"].get<int>();
        }
        if (config.contains("batch_time_us")) {
            this->batch_time_us = config["batch_time_us"].get<uint32_t>();
        }
        if (config.contains("min_batch_size")) {
            this->min_batch_size = config["min_batch_size"].get<uint32_t>();
        }
        if (config.contains("max_batch_size")) {
            this->max_batch_size = config["max_batch_size"].get<uint32_t>();
        }
        if (config.contains("max_process_batch_size")) {
            this->max_process_batch_size = config["max_process_batch_size"].get<uint32_t>();
        }
        // Currenlty only support multithreading for hnswlib search, as GPU flat search requires cuda streams for host side parallelism
        if (config.contains("num_threads") && static_cast<GroupedEmbeddingsForSearch::SearchType>(this->faiss_search_type) == GroupedEmbeddingsForSearch::SearchType::HnswlibCpuSearch) {
            this->num_threads = config["num_threads"].get<int>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
        dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
    }
    this->cluster_search_index = std::make_shared<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim, hnsw_config);
    this->start_threads(typed_ctxt);
}

void ClustersSearchOCDPO::shutdown() {
    // Clean up index resources
    cluster_search_index->reset();  
    for (auto& cluster_search_thread : cluster_search_threads) {
        if (cluster_search_thread) {
            cluster_search_thread->signal_stop();
            cluster_search_thread->join();
        }
    }

    if (batch_thread) {
        batch_thread->signal_stop();
        batch_thread->join();
    }
}

} // namespace cascade
} // namespace derecho
