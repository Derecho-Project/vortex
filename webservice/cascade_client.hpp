#pragma once

#include <cascade/service_client_api.hpp>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <queue>
#include <iostream>
#include <future>
#include <tuple>
#include <atomic>
#include <unordered_map>
#include <shared_mutex>

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

using query_id_t = uint64_t; 
using queued_query_t = std::tuple<query_id_t,std::string,const float*>;

class VortexCascadeClient {
    /*
     * This thread is responsible for batching queries and sending them to the first UDL in the pipeline (using trigger_put).
     */
    class ClientThread {
    private:
        std::thread real_thread;
        ServiceClientAPI& capi = ServiceClientAPI::get_service_client();
        uint64_t node_id = capi.get_my_id();
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

        std::queue<Blob> to_process;
        VortexCascadeClient* vortex;

        void main_loop();

    public:
        NotificationThread(VortexCascadeClient* vortex);
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
    uint64_t my_id = capi.get_my_id();
    ClientThread *client_thread;
    std::deque<NotificationThread> notification_threads;
    uint64_t emb_dim = 1024;
    uint64_t num_result_threads = 1;
    uint64_t next_thread = 0;

    std::mutex query_id_mtx;
    uint64_t query_count = 0;
    query_id_t next_query_id();

    std::shared_mutex result_mutex;
    std::unordered_map<query_id_t,std::promise<std::vector<std::string>>> result;
    void result_received(nlohmann::json &result_json);

    public:

    VortexCascadeClient();
    ~VortexCascadeClient();
    
    void setup(uint64_t batch_min_size,uint64_t batch_max_size,uint64_t batch_time_us,uint64_t emb_dim,uint64_t num_result_threads);
    std::future<std::vector<std::string>> query(const std::string& query,const float* query_emb);
};

