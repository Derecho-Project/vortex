#pragma once

#include <algorithm>
#include <memory>
#include <map>
#include <iostream>
#include <tuple>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>

#include "serialize_utils.hpp"

namespace derecho {
namespace cascade {

#define MY_UUID "11a3c123-3300-31ac-1866-0003ac330000"
#define MY_DESC "UDL to aggregate the knn search results for each query from the clusters and run LLM with the query and its top_k closest docs."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class AggGenOCDPO : public DefaultOffCriticalDataPathObserver {

private:
    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

    class ProcessThread {
        private:
            uint64_t my_thread_id;
            AggGenOCDPO* parent;
            std::thread real_thread;
            
            bool running = false;
            std::mutex thread_mtx;
            std::condition_variable thread_signal;

            std::queue<std::shared_ptr<ClusterSearchResult>> pending_results; // results to be processed
            std::unordered_map<query_id_t,std::unique_ptr<ClusterSearchResultsAggregate>> results_aggregate; // cluster search results aggregate of each query

            void process_result(std::shared_ptr<ClusterSearchResult> result);
            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            ProcessThread(uint64_t thread_id, AggGenOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
            void push_result(std::shared_ptr<ClusterSearchResult> result);
    };

    /***
     * This thread gathers aggregate results for each client, batch them and notify the client
     */
    class BatchingThread {
        private:
            uint64_t my_thread_id;
            AggGenOCDPO* parent;
            std::thread real_thread;
            bool running = false;

            std::unordered_map<uint32_t,std::unique_ptr<std::vector<std::unique_ptr<ClusterSearchResultsAggregate>>>> client_queue; // a queue for each client
            std::condition_variable_any client_queue_cv;
            std::mutex client_queue_mutex;

            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            BatchingThread(uint64_t thread_id, AggGenOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
            void push_aggregate_results(std::unique_ptr<ClusterSearchResultsAggregate> aggregate);
    };

    int top_k = 5; // final top K results to use for LLM
    int top_num_centroids = 4; // number of top K clusters need to wait to gather for each query
    int min_batch_size = 1;
    int max_batch_size = 10;
    int batch_time_us = 100;
    int my_id;
    uint64_t num_threads = 1; // number of threads to process the cluster search results

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    std::unordered_map<uint64_t,std::unordered_map<long,long>> cluster_doc_table; // maps the local cluster doc ids to the global ids
    bool load_doc_table(DefaultCascadeContextType* typed_ctxt, uint64_t cluster_id);

    /* 
     * The code below should be moved to the new UDL4, which will be responsible for getting the documents and calling/running the LLM
     *
    int retrieve_docs = true; // 0: not retrieve, 1: retrieve
    
    std::shared_mutex doc_cache_mutex;
    // TODO: could use a more efficient way to store the doc_contents cache
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_tables; // cluster_id -> emb_index -> pathname
    std::unordered_map<int, std::unordered_map<long, std::string>> doc_contents; // {cluster_id0:{ emb_index0: doc content0, ...}, cluster_id1:{...}, ...}
    */

    /*** Helper function to load the doc table for a given cluster_id
     *   Use the doc_table could find the pathname of a text document that corresponds to a given cluster's embedding index
     */
    //bool load_doc_table(DefaultCascadeContextType* typed_ctxt, int cluster_id);
    /*** Helper function to get a doc for a given cluster_id and emb_index. Used by get_topk_docs
     *   First check if the doc is in the cache, if not, cache it after retrieving it from cascade
     */
    //bool get_doc(DefaultCascadeContextType* typed_ctxt, int cluster_id, long emb_index, std::string& res_doc);

public:
    std::vector<std::unique_ptr<ProcessThread>> process_threads;
    std::unique_ptr<BatchingThread> batch_thread;

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<AggGenOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }
    void start_threads(DefaultCascadeContextType* typed_ctxt);
    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);
    void shutdown() ;
};

std::shared_ptr<OffCriticalDataPathObserver> AggGenOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    AggGenOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(ICascadeContext* ctxt, 
                                                        const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->set_config(typed_ctxt, config);
    return AggGenOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    std::static_pointer_cast<AggGenOCDPO>(AggGenOCDPO::get())->shutdown();
    return;
}

}  // namespace cascade
}  // namespace derecho

