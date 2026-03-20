
#include "cascade_client.hpp"

VortexCascadeClient::VortexCascadeClient(){
}

VortexCascadeClient::~VortexCascadeClient(){
    client_thread->signal_stop();
    client_thread->join();

    for(auto &t : notification_threads){
        t.signal_stop();
    }
    
    for(auto &t : notification_threads){
        t.join();
    }
}

void VortexCascadeClient::setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads){
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

query_id_t VortexCascadeClient::next_query_id(){
    std::unique_lock<std::mutex> lock(query_id_mtx);
    query_id_t query_id = (my_id << 48) | query_count;
    query_count++;
    return query_id;
}

std::future<std::shared_ptr<VortexANNResult>> VortexCascadeClient::query_ann(std::shared_ptr<float> query_emb){
    query_id_t query_id = VortexCascadeClient::next_query_id();

    std::shared_ptr<std::string> query = std::make_shared<std::string>(); // empty query text
    queued_query_t new_query(query_id,my_id,query_emb,query);

    std::unique_lock<std::shared_mutex> lock(result_mutex);
    auto& prom = result[query_id];
    lock.unlock();

    client_thread->push_query(new_query);
    return prom.get_future();
}

void VortexCascadeClient::ann_result_received(std::shared_ptr<VortexANNResult> res){
    query_id_t query_id = res->get_query_id();

    std::shared_lock<std::shared_mutex> lock(result_mutex);
    auto& prom = result[query_id];
    lock.unlock();

    prom.set_value(res);
}

// client thread methods

VortexCascadeClient::ClientThread::ClientThread(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim){
    this->batch_min_size = batch_min_size;
    this->batch_max_size = batch_max_size;
    this->batch_time_us = batch_time_us;
    this->emb_dim = emb_dim;
}

void VortexCascadeClient::ClientThread::push_query(queued_query_t &queued_query){
    std::unique_lock<std::mutex> lock(thread_mtx);
    query_queue.push(queued_query);
    thread_signal.notify_all();
}

void VortexCascadeClient::ClientThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexCascadeClient::ClientThread::main_loop(){
    if(!running) return;
    
    // thread main loop
    EmbeddingQueryBatcher batcher(emb_dim,batch_max_size);
    auto wait_start = std::chrono::steady_clock::now();
    auto batch_time = std::chrono::microseconds(batch_time_us);
    uint64_t batch_id = 0;
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

            // trigger put
            capi.trigger_put(obj);

            batch_id++;
            batcher.reset();
        }
    }
}

// notification thread methods

VortexCascadeClient::NotificationThread::NotificationThread(VortexCascadeClient* vortex){
    this->vortex = vortex;
}

void VortexCascadeClient::NotificationThread::push_result(const Blob& result){
    std::shared_ptr<uint8_t> buffer(new uint8_t[result.size]);
    std::memcpy(buffer.get(),result.bytes,result.size);

    std::unique_lock<std::mutex> lock(thread_mtx);
    to_process.emplace(buffer,result.size);
    thread_signal.notify_all();
}

void VortexCascadeClient::NotificationThread::signal_stop(){
    std::unique_lock<std::mutex> lock(thread_mtx);
    running = false;
    thread_signal.notify_all();
}

void VortexCascadeClient::NotificationThread::main_loop(){
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

