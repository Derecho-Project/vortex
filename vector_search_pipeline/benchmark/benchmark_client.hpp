#pragma once

#include <cascade/service_client_api.hpp>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <iostream>
#include <tuple>
#include <atomic>
#include <unordered_map>
#include <shared_mutex>
#include "../vortex_udls/serialize_utils.hpp"

using namespace derecho::cascade;

#define UDL1_PATH "/rag/emb/centroids_search"
#define UDL2_PATH "/rag/emb/clusters_search"
#define UDL3_PATH "/rag/generate/agg"
#define UDL1_TIMESTAMP_FILE "udl1.dat"
#define UDL2_TIMESTAMP_FILE "udl2.dat"
#define UDL3_TIMESTAMP_FILE "udl3.dat"
#define UDL1_SUBGROUP_INDEX 0
#define UDL2_SUBGROUP_INDEX 1
#define UDL3_SUBGROUP_INDEX 2
#define UDLS_SUBGROUP_TYPE VolatileCascadeStoreWithStringKey 

#define VORTEX_CLIENT_MAX_WAIT_TIME 60

class VortexBenchmarkClient {
    /*
     * This thread is responsible for batching queries and sending them to the first UDL in the pipeline (using trigger_put).
     */
    class ClientThread {
    private:
        std::thread real_thread;
        ServiceClientAPI& capi = ServiceClientAPI::get_service_client();
        uint32_t node_id = capi.get_my_id();
        uint64_t batch_min_size = 0;
        uint64_t batch_max_size = 5;
        uint64_t batch_time_us = 500;
        uint64_t emb_dim = 1024;

        bool running = false;
        std::mutex thread_mtx;
        std::condition_variable thread_signal;

        std::queue<queued_query_t> query_queue;

        void main_loop();

    public:
        ClientThread(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim);
        void push_query(queued_query_t &queued_query);
        void signal_stop();
        std::unordered_map<uint64_t,uint64_t> batch_size;
        uint64_t batch_id = 0;

        inline void start(){
            running = true;
            real_thread = std::thread(&ClientThread::main_loop,this);
        }

        inline void join(){
            real_thread.join();
        }
    };

    /*
     * This thread is responsible for deserializing results received by the last UDL in the pipeline.
     */
    class NotificationThread {
    private:
        std::thread real_thread;
        bool running = false;
        std::mutex thread_mtx;
        std::condition_variable thread_signal;

        std::queue<std::pair<std::shared_ptr<uint8_t>,uint64_t>> to_process;
        VortexBenchmarkClient* vortex;

        void main_loop();

    public:
        NotificationThread(VortexBenchmarkClient* vortex);
        void push_result(const Blob& result);
        void signal_stop();

        inline void start(){
            running = true;
            real_thread = std::thread(&NotificationThread::main_loop,this);
        }

        inline void join(){
            real_thread.join();
        }
    };

    ServiceClientAPI& capi = ServiceClientAPI::get_service_client();
    uint32_t my_id = capi.get_my_id();
    ClientThread *client_thread;
    std::deque<NotificationThread> notification_threads;
    uint64_t emb_dim = 1024;
    uint64_t num_result_threads = 1;
    uint64_t next_thread = 0;

    std::unordered_map<query_id_t,std::chrono::steady_clock::time_point> query_send_time;
    std::unordered_map<query_id_t,std::chrono::steady_clock::time_point> query_result_time;
    uint64_t skip;

    std::mutex query_id_mtx;
    uint64_t query_count = 0;
    query_id_t next_query_id();

    std::atomic<uint64_t> result_count = 0;
    std::shared_mutex result_mutex;
    std::unordered_map<query_id_t,std::shared_ptr<VortexANNResult>> result;
    void ann_result_received(std::shared_ptr<VortexANNResult> result);

    public:

    VortexBenchmarkClient(uint64_t skip = 0);
    ~VortexBenchmarkClient();
    
    void setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads);
   
    uint64_t query(std::shared_ptr<std::string> query,std::shared_ptr<float> query_emb);
    void wait_results();
    std::shared_ptr<VortexANNResult> get_result(query_id_t query_id);
    void dump_timestamps(bool dump_remote = true);
};

