
#include "benchmark_client.hpp"

VortexBenchmarkClient::VortexBenchmarkClient(uint64_t skip){
    this->skip = skip;
}

VortexBenchmarkClient::~VortexBenchmarkClient(){
    client_thread->signal_stop();
    client_thread->join();

    for(auto &t : notification_threads){
        t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }

    // print e2e performance statistics (discarding the first 'skip' queries)
    std::vector<query_id_t> queries;
    for (const auto& [query_id, send_time] : query_send_time){
        if(query_result_time.count(query_id) == 0) continue;
        queries.push_back(query_id);
    }
    std::sort(queries.begin(),queries.end());

    uint64_t num_queries = queries.size();
    std::vector<double> latencies;
    std::chrono::steady_clock::time_point first = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last;
    double sum = 0.0;
    for(uint64_t i=skip; i<num_queries; i++){
        auto query_id = queries[i];
        auto& sent = query_send_time[query_id];
        auto& received = query_result_time[query_id];
        std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(received - sent);

        double lat = static_cast<double>(elapsed.count()) / 1000.0;
        latencies.push_back(lat);
        sum += lat;

        first = std::min(first,sent);
        last = std::max(last,received);
    }

    std::chrono::microseconds total_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(last - first);
    double total_time = static_cast<double>(total_elapsed.count()) / 1000000.0;
    double throughput = (num_queries-skip) / total_time;
    std::sort(latencies.begin(),latencies.end());
    double avg = sum / latencies.size();
    double min = latencies.front();
    double max = latencies.back();
    auto median = latencies[latencies.size()/2];
    auto p95 = latencies[(uint64_t)(latencies.size()*0.95)];

    std::cout << "Throughput: " << throughput << " queries/s" << " (" << num_queries-skip << " queries in " << total_time << " seconds)" << std::endl;
    std::cout << "E2E latency:" << std::endl;
    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;

    // print batching statistics (discarding the first 10%)
    std::cout << "batching statistics:" << std::endl;
    std::vector<uint64_t> values;
    values.reserve(client_thread->batch_size.size());
    uint64_t start = (uint64_t)(client_thread->batch_id * 0.1);
    sum = 0.0;
    for(const auto& [batch_id, sz] : client_thread->batch_size){
        if(batch_id < start) continue;
        values.push_back(sz);
        sum += sz;
    }

    avg = sum / values.size();
    std::sort(values.begin(),values.end());
    min = values.front();
    max = values.back();
    median = values[values.size()/2];
    p95 = values[(uint64_t)(values.size()*0.95)];

    std::cout << "  avg: " << avg << std::endl;
    std::cout << "  median: " << median << std::endl;
    std::cout << "  min: " << min << std::endl;
    std::cout << "  max: " << max << std::endl;
    std::cout << "  p95: " << p95 << std::endl;
}

void VortexBenchmarkClient::setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads){
    this->emb_dim = emb_dim;
    this->num_result_threads = num_result_threads;

    // Prepare for the notification by creating object pool for results to store
    std::string result_pool_name = "/rag/results/" + std::to_string(this->my_id);
    std::cout << "  creating object pool for receiving results: " << result_pool_name << std::endl;
    auto res = capi.template create_object_pool<UDLS_SUBGROUP_TYPE>(result_pool_name,UDL3_SUBGROUP_INDEX,HASH,{});
    for (auto& reply_future:res.get()) {
        reply_future.second.get(); // wait for the object pool to be created
    }

    // Register notification for this object pool
    std::cout << "  registering notification handler ... " << result_pool_name << std::endl;
    bool ret = capi.register_notification_handler(
            [&](const Blob& result){
                notification_threads[next_thread].push_result(result);
                next_thread = (next_thread + 1) % this->num_result_threads;
                return true;
            }, result_pool_name);

    // Establish connections to all server nodes running UDL3
    auto shards = capi.get_subgroup_members<UDLS_SUBGROUP_TYPE>(UDL3_SUBGROUP_INDEX);
    std::cout << "  pre-establishing connections with all nodes in " << shards.size() << " shards ..." << std::endl;
    ObjectWithStringKey obj;
    obj.key = "establish_connection";
    std::string control_value = "establish";
    obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
    uint32_t i = 0;
    for(auto& shard : shards){
        for(int j=0;j<shard.size();j++){
            // each iteration reaches a different node in the shard due to the round robin policy
            auto res = capi.template put<UDLS_SUBGROUP_TYPE>(obj, UDL3_SUBGROUP_INDEX, i, true);
            for (auto& reply_future:res.get()) {
                reply_future.second.get(); // wait for the object has been put
            }
        }
        i++;
    }

    // start notification threads
    for(uint64_t i=0;i<num_result_threads;i++){
        notification_threads.emplace_back(this);
    }

    for(auto &t : notification_threads){
        t.start();
    }

    // start client thread
    client_thread = new ClientThread(batch_min_size,batch_max_size,batch_time_us,emb_dim);
    client_thread->start();
}

query_id_t VortexBenchmarkClient::next_query_id(){
    std::unique_lock<std::mutex> lock(query_id_mtx);
    query_id_t query_id = (static_cast<uint64_t>(my_id) << 48) | query_count;
    query_count++;
    return query_id;
}

uint64_t VortexBenchmarkClient::query(std::shared_ptr<std::string> query,std::shared_ptr<float> query_emb){
    query_id_t query_id = VortexBenchmarkClient::next_query_id();
    queued_query_t new_query(query_id,my_id,query_emb,query);
    query_send_time[query_id] = std::chrono::steady_clock::now();
    client_thread->push_query(new_query);
    return query_id;
}

void VortexBenchmarkClient::wait_results(){
    std::cout << "  received " << result_count << std::endl;
    uint64_t wait_time = 0;
    while(result_count < query_count){
        if(wait_time >= VORTEX_CLIENT_MAX_WAIT_TIME){
            std::cout << "  waited more than " << VORTEX_CLIENT_MAX_WAIT_TIME << " seconds, stopping ..." << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "  received " << result_count << std::endl;
        wait_time += 2;
    }
}

void VortexBenchmarkClient::dump_timestamps(bool dump_remote){
    TimestampLogger::flush("client" + std::to_string(my_id) + ".dat");

    if(dump_remote){
        capi.dump_timestamp(UDL1_TIMESTAMP_FILE,UDL1_PATH);
        capi.dump_timestamp(UDL2_TIMESTAMP_FILE,UDL2_PATH);
        capi.dump_timestamp(UDL3_TIMESTAMP_FILE,UDL3_PATH);
    }
}

void VortexBenchmarkClient::ann_result_received(std::shared_ptr<VortexANNResult> res){
    query_id_t query_id = res->get_query_id();
    std::unique_lock<std::shared_mutex> lock(result_mutex);
    result.emplace(query_id,res);
    result_count++;
    
    query_result_time[query_id] = std::chrono::steady_clock::now();
    
    TimestampLogger::log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,my_id,query_id,0); // TODO revise
}

std::shared_ptr<VortexANNResult> VortexBenchmarkClient::get_result(query_id_t query_id){
    std::shared_lock<std::shared_mutex> lock(result_mutex);
    return result[query_id];
}

// client thread methods

VortexBenchmarkClient::ClientThread::ClientThread(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim){
    this->batch_min_size = batch_min_size;
    this->batch_max_size = batch_max_size;
    this->batch_time_us = batch_time_us;
    this->emb_dim = emb_dim;
}

void VortexBenchmarkClient::ClientThread::push_query(queued_query_t &queued_query){
    std::unique_lock<std::mutex> lock(thread_mtx);
    query_queue.push(queued_query);
    thread_signal.notify_all();
}

void VortexBenchmarkClient::ClientThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexBenchmarkClient::ClientThread::main_loop(){
    if(!running) return;
    
    // thread main loop
    EmbeddingQueryBatcher batcher(emb_dim,batch_max_size);
    std::vector<uint64_t> id_list; // for logging purposes
    id_list.reserve(batch_max_size);
    auto wait_start = std::chrono::steady_clock::now();
    auto batch_time = std::chrono::microseconds(batch_time_us);
    while(true){
        std::unique_lock<std::mutex> lock(thread_mtx);
        if(query_queue.empty()){
            thread_signal.wait_for(lock,batch_time);
        }

        if(!running) break;

        uint64_t send_count = 0;
        uint64_t queued_count = query_queue.size();
        auto now = std::chrono::steady_clock::now();

        if((queued_count >= batch_min_size) || ((now-wait_start) >= batch_time)){
            send_count = std::min(queued_count,batch_max_size);
            wait_start = now;

            // copy out queries
            for(uint64_t i=0;i<send_count;i++){
                queued_query_t &query = query_queue.front();
                batcher.add_query(query);
                id_list.push_back(std::get<0>(query));
                query_queue.pop();
            }
        }

        lock.unlock();

        // now we are outside the locked region (i.e the client can continue adding queries to the queue): build object and call trigger_put
        if(send_count > 0){
            batcher.serialize();

            ObjectWithStringKey obj;
            obj.key = UDL1_PATH "/" + std::to_string(node_id) + "_" + std::to_string(batch_id); // batch_id is important for randomizing the shard that gets each batch
            obj.blob = std::move(*batcher.get_blob());

            for(uint64_t i=0;i<send_count;i++){
                auto query_id = id_list[i];
                TimestampLogger::log(LOG_TAG_QUERIES_SENDING_START,node_id,batch_id,i); // TODO update with just query_id
            }

            // trigger put
            capi.trigger_put(obj);
            
            for(uint64_t i=0;i<send_count;i++){
                auto query_id = id_list[i];
                TimestampLogger::log(LOG_TAG_QUERIES_SENDING_END,node_id,batch_id,i); // TODO update with just query_id
            }

            batch_size[batch_id] = send_count; // for statistics
            batch_id++;

            batcher.reset();
            id_list.clear();
        }
    }
}

// notification thread methods

VortexBenchmarkClient::NotificationThread::NotificationThread(VortexBenchmarkClient* vortex){
    this->vortex = vortex;
}

void VortexBenchmarkClient::NotificationThread::push_result(const Blob& result){
    std::shared_ptr<uint8_t> buffer(new uint8_t[result.size]);
    std::memcpy(buffer.get(),result.bytes,result.size);

    std::unique_lock<std::mutex> lock(thread_mtx);
    to_process.emplace(buffer,result.size);
    thread_signal.notify_all();
}

void VortexBenchmarkClient::NotificationThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexBenchmarkClient::NotificationThread::main_loop(){
    if(!running) return;
    
    // thread main loop
    while(true){
        std::unique_lock<std::mutex> lock(thread_mtx);
        if(to_process.empty()){
            thread_signal.wait(lock);
        }

        if(!running) break;

        auto pending = to_process.front();
        to_process.pop();
        lock.unlock();
        
        if (pending.second == 0) {
             std::cerr << "Error: empty result blob." << std::endl;
             continue;
        }

        ClientNotificationManager manager(pending.first,pending.second);
        for(auto& ann_result : manager.get_results()){
            vortex->ann_result_received(ann_result);
        }
    }
}

