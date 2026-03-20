#include "aggregate_udl.hpp"

namespace derecho {
namespace cascade {

// ProcessThread

AggGenOCDPO::ProcessThread::ProcessThread(uint64_t thread_id, AggGenOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}

void AggGenOCDPO::ProcessThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&ProcessThread::main_loop, this, typed_ctxt);
}

void AggGenOCDPO::ProcessThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void AggGenOCDPO::ProcessThread::signal_stop() {
    running = false;
    thread_signal.notify_all(); 
}

void AggGenOCDPO::ProcessThread::push_result(std::shared_ptr<ClusterSearchResult> result) {
    std::lock_guard<std::mutex> lock(thread_mtx);
    pending_results.push(result);
    thread_signal.notify_all();
}

void AggGenOCDPO::ProcessThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    while (running) {
        std::unique_lock<std::mutex> lock(thread_mtx);
        if (this->pending_results.empty()){
            thread_signal.wait(lock);
        }

        if (!running) break;

        if(this->pending_results.empty()) continue;

        std::shared_ptr<ClusterSearchResult> result = this->pending_results.front();
        this->pending_results.pop();

        lock.unlock();

        // process pending result
        process_result(result);
    }
}

void AggGenOCDPO::ProcessThread::process_result(std::shared_ptr<ClusterSearchResult> result){
    // add results together with results received before
    query_id_t query_id = result->get_query_id();
    if(results_aggregate.count(query_id) == 0){
        results_aggregate[query_id] = std::make_unique<ClusterSearchResultsAggregate>(result,parent->top_num_centroids, parent->top_k,&parent->cluster_doc_table);
    } else {
        results_aggregate[query_id]->add_result(result);
    }

    // in case all results were received, send it to the batching thread so the client can be notified
    if(results_aggregate[query_id]->all_results_received()){
        parent->batch_thread->push_aggregate_results(std::move(results_aggregate[query_id]));
        results_aggregate.erase(query_id);
    }
}

// BatchingThread

AggGenOCDPO::BatchingThread::BatchingThread(uint64_t thread_id, AggGenOCDPO* parent_udl)
    : my_thread_id(thread_id), parent(parent_udl), running(false) {}

void AggGenOCDPO::BatchingThread::start(DefaultCascadeContextType* typed_ctxt) {
    running = true;
    real_thread = std::thread(&BatchingThread::main_loop, this, typed_ctxt);
}

void AggGenOCDPO::BatchingThread::join() {
    if (real_thread.joinable()) {
        real_thread.join();
    }
}

void AggGenOCDPO::BatchingThread::signal_stop() {
    std::lock_guard<std::mutex> lock(client_queue_mutex);
    running = false;
    client_queue_cv.notify_all();
}

void AggGenOCDPO::BatchingThread::push_aggregate_results(std::unique_ptr<ClusterSearchResultsAggregate> aggregate){
    uint32_t client_id = aggregate->get_client_id();
    std::unique_lock<std::mutex> lock(client_queue_mutex);

    if(client_queue.count(client_id) == 0){
        client_queue[client_id] = std::make_unique<std::vector<std::unique_ptr<ClusterSearchResultsAggregate>>>();
        client_queue[client_id]->reserve(parent->max_batch_size);
    }

    client_queue[client_id]->push_back(std::move(aggregate));

    client_queue_cv.notify_all();
}

void AggGenOCDPO::BatchingThread::main_loop(DefaultCascadeContextType* typed_ctxt) {
    // TODO timestamp logging in this method should be revisited

    std::unique_lock<std::mutex> lock(client_queue_mutex, std::defer_lock);
    std::unordered_map<uint32_t,std::chrono::steady_clock::time_point> wait_time;
    auto batch_time = std::chrono::microseconds(parent->batch_time_us);
    while (running) {
        lock.lock();
        bool empty = true;
        for(auto& item : client_queue){
            if(!(item.second->empty())){
                empty = false;
                break;
            }
        }

        if(empty){
            client_queue_cv.wait_for(lock,batch_time);
        }

        if (!running) break;

        // move queue pointers out of the map and replace with empty vectors
        std::unordered_map<uint32_t,std::unique_ptr<std::vector<std::unique_ptr<ClusterSearchResultsAggregate>>>> to_send;
        auto now = std::chrono::steady_clock::now();
        for(auto& item : client_queue){
            if(wait_time.count(item.first) == 0){
                wait_time[item.first] = now;
            }

            if((item.second->size() >= parent->min_batch_size) || ((now-wait_time[item.first]) >= batch_time)){
                to_send[item.first] = std::move(item.second);
                item.second = std::make_unique<std::vector<std::unique_ptr<ClusterSearchResultsAggregate>>>();
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

                ClientNotificationBatcher batcher(parent->top_k,batch_size,false);
                for(uint64_t i=num_sent;i<(num_sent+batch_size);i++){
                    batcher.add_aggregate(std::move(item.second->at(i)));
                }
                batcher.serialize();

                // notify client
                try {
                    std::string notification_pathname = "/rag/results/" + std::to_string(item.first);
                    typed_ctxt->get_service_client_ref().notify(*(batcher.get_blob()),notification_pathname,item.first);
                    dbg_default_trace("[AggregateGenUDL] echo back to node {}", item.first);
                } catch (derecho::derecho_exception& ex) {
                    std::cerr << "[AGGnotification ocdpo]: exception on notification:" << ex.what() << std::endl;
                    dbg_default_error("[AGGnotification ocdpo]: exception on notification:{}", ex.what());
                }

                num_sent += batch_size;
            }
        }
    }
}

void AggGenOCDPO::ocdpo_handler(const node_id_t sender, 
                                const std::string& object_pool_pathname, 
                                const std::string& key_string, 
                                const ObjectWithStringKey& object, 
                                const emit_func_t& emit, 
                                DefaultCascadeContextType* typed_ctxt, 
                                uint32_t worker_id) {

    uint64_t cluster_id = parse_cluster_id_udl3(key_string);
    if(cluster_doc_table.count(cluster_id) == 0){
        if(!load_doc_table(typed_ctxt,cluster_id)){
            std::cerr << "Error: doc table could not be loaded for cluster " << cluster_id << std::endl;
            dbg_default_error("doc table could not be loaded at AggGenOCDPO");
            return;
        }
    }

    TimestampLogger::log(LOG_TAG_AGG_UDL_START,my_id,0,sender); // TODO revise
    
    std::unique_ptr<ClusterSearchResultBatchManager> batch_manager = std::make_unique<ClusterSearchResultBatchManager>(object.blob.bytes,object.blob.size,cluster_id);
    
    TimestampLogger::log(LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE, my_id, 0, sender); // TODO revise
   
    for(auto& result : batch_manager->get_results()){ 
        query_id_t query_id = result->get_query_id();
        uint64_t to_thread = query_id % num_threads;
        process_threads[to_thread]->push_result(result);
    }

    // TODO: add a logger here after push?
}

void AggGenOCDPO::start_threads(DefaultCascadeContextType* typed_ctxt) {
    this->batch_thread = std::make_unique<BatchingThread>(this->my_id, this);
    this->batch_thread->start(typed_ctxt);

    for (int thread_id = 0; thread_id < this->num_threads; thread_id++) {
        process_threads.emplace_back(std::make_unique<ProcessThread>(static_cast<uint64_t>(thread_id), this));
    }
    for (auto& process_thread : process_threads) {
        process_thread->start(typed_ctxt);
    }
}

void AggGenOCDPO::set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config) {
    this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
    try{
        if (config.contains("top_num_centroids")) this->top_num_centroids = config["top_num_centroids"].get<int>();
        if (config.contains("final_top_k")) this->top_k = config["final_top_k"].get<int>();
        if (config.contains("batch_time_us")) this->batch_time_us = config["batch_time_us"].get<int>();
        if (config.contains("min_batch_size")) this->min_batch_size = config["min_batch_size"].get<int>();
        if (config.contains("max_batch_size")) this->max_batch_size = config["max_batch_size"].get<int>();
        if (config.contains("num_threads")) this->num_threads = config["num_threads"].get<int>();
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config" << std::endl;
        dbg_default_error("Failed to convert top_num_centroids, top_k, include_llm, or retrieve_docs from config, at clusters_search_udl.");
    } 
    this->start_threads(typed_ctxt);
}  

void AggGenOCDPO::shutdown() {
    for (auto& process_thread : process_threads) {
        if (process_thread) {
            process_thread->signal_stop();
            process_thread->join();
        }
    }

    if (batch_thread) {
        batch_thread->signal_stop();
        batch_thread->join();
    }
}

bool AggGenOCDPO::load_doc_table(DefaultCascadeContextType* typed_ctxt, uint64_t cluster_id) {
    if (cluster_doc_table.count(cluster_id) > 0){
        return true;
    }

    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START, my_id, 0, cluster_id);
    // check the keys for this grouped embedding objects stored in cascade
    //    because of the message size, the map for one cluster may split into multiple chunks stored in Cascade
    bool stable = 1; 
    persistent::version_t version = CURRENT_VERSION;
    std::string table_prefix = "/rag/doc/emb_doc_map/cluster" + std::to_string(cluster_id);
    auto keys_future = typed_ctxt->get_service_client_ref().list_keys(version, stable, table_prefix);
    std::vector<std::string> map_obj_keys = typed_ctxt->get_service_client_ref().wait_list_keys(keys_future);
    if (map_obj_keys.empty()) {
        std::cerr << "Error: " << table_prefix <<" has no emb_doc_map object found in the KV store" << std::endl;
        dbg_default_error("[{}]at {}, Failed to find object prefix {} in the KV store.", gettid(), __func__, table_prefix);
        return -1;
    }

    std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filtered_keys = filter_exact_matched_keys(map_obj_keys, table_prefix);
    // get the doc table objects and save to local cache
    while(!filtered_keys.empty()){
        std::string map_obj_key = filtered_keys.top();
        filtered_keys.pop();
        auto get_query_results = typed_ctxt->get_service_client_ref().get(map_obj_key);
        auto& reply = get_query_results.get().begin()->second.get();
        if (reply.blob.size == 0) {
            std::cerr << "Error: failed to get the doc table for key=" << map_obj_key << std::endl;
            dbg_default_error("Failed to get the doc table for key={}.", map_obj_key);
            return false;
        }
        char* json_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
        std::string json_str(json_data, reply.blob.size);
        try{
            nlohmann::json doc_table_json = nlohmann::json::parse(json_str);
            for (const auto& [emb_index, pathname] : doc_table_json.items()) {
                this->cluster_doc_table[cluster_id][std::stol(emb_index)] = pathname.get<long>();
            }
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "Error: load_doc_table JSON parse error: " << e.what() << std::endl;
            dbg_default_error("{}, JSON parse error: {}", __func__, e.what());
            return false;
        }
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END, my_id, 0, cluster_id);
    return true;
}

/* 
 * The code below should be moved to the new UDL4, which will be responsible for getting the documents and calling/running the LLM
 *
bool AggGenOCDPO::get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc) {
    std::shared_lock<std::shared_mutex> read_lock(doc_cache_mutex);
    if (doc_contents.find(cluster_id) != doc_contents.end()) {
        if (doc_contents[cluster_id].find(emb_index) != doc_contents[cluster_id].end()) {
            res_doc = doc_contents[cluster_id][emb_index];
            return true;
        }
    }
    read_lock.unlock();
    // doc not found in the cache, load the emb_index->doc table and content, and cache it
    std::unique_lock<std::shared_mutex> write_lock(doc_cache_mutex);
    bool loaded_doc_table = load_doc_table(typed_ctxt, cluster_id);
    if (!loaded_doc_table) {
        dbg_default_error("Failed to load the doc table for cluster_id={}.", cluster_id);
        return false;
    }
    if (doc_tables[cluster_id].find(emb_index) == doc_tables[cluster_id].end()) {
        std::cerr << "Error: failed to find the doc pathname for cluster_id=" << cluster_id << " and emb_id=" << emb_index << std::endl;
        dbg_default_error("Failed to find the doc pathname for cluster_id={} and emb_id={}, query={}.", cluster_id, emb_index);
        return false;
    }
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_START, this->my_id, emb_index, cluster_id);
    auto& pathname = doc_tables[cluster_id][emb_index];
    if(!retrieve_docs){
        res_doc = pathname;
        return true;
    }
    auto get_doc_results = typed_ctxt->get_service_client_ref().get(pathname);
    auto& reply = get_doc_results.get().begin()->second.get();
    if (reply.blob.size == 0) {
        std::cerr << "Error: failed to cascade get the doc content for pathname=" << pathname << std::endl;
        dbg_default_error("Failed to cascade get the doc content for pathname={}.", pathname);
        return false;
    }
    // parse the reply.blob.bytes to std::string
    char* doc_data = const_cast<char*>(reinterpret_cast<const char*>(reply.blob.bytes));
    std::string doc_str(doc_data, reply.blob.size);  
    this->doc_contents[cluster_id][emb_index] = doc_str;
    res_doc = this->doc_contents[cluster_id][emb_index];
    TimestampLogger::log(LOG_TAG_AGG_UDL_LOAD_DOC_END, this->my_id, emb_index, cluster_id);
    return true;
}
*/

}  // namespace cascade
}  // namespace derecho
