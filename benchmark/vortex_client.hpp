#pragma once
#include <cascade/service_client_api.hpp>
#include <chrono>
#include <filesystem> 
#include <iostream>
#include <unistd.h>  
#include <vector>
#include "../vortex_udls/rag_utils.hpp"

using namespace derecho::cascade;
// #define EMBEDDING_DIM 1024
#define MAX_NUM_EMB_PER_OBJ 200  // maximum number of embeddings could be batched per object
#define VORTEX_SUBGROUP_INDEX 0
#define AGG_SUBGROUP_INDEX 0
#define QUERY_FILENAME "query.csv"
#define QUERY_EMB_FILENAME "query_emb.csv"
#define GROUNDTRUTH_FILENAME "groundtruth.csv"


inline bool is_in_topk(const std::vector<std::string>& groundtruth, const std::string& target, int k) {
     k = std::min(k, (int)groundtruth.size());
     auto it = std::find(groundtruth.begin(), groundtruth.begin() + k, target);
     return it != (groundtruth.begin() + k);
}

class VortexPerfClient{
     int my_node_id;

     int num_queries = 0;
     int batch_size = 0;
     // int query_interval = 100000; // default interval between query is 1 second
     int query_interval = 50000;
     int embedding_dim = 1024;

     // Use vector since one query may be reuse for multiple times among different batches. 
     // sent_queries: query_text -> [(batch_id,query_id), ...]
     // Note, we are currently not handling the case if there are duplicated queries in the same batch. 
     // this should be done at client side, here. To avoid send batch with duplication but keep track of it to return to the RESTFUL clients that have the same query.
     std::unordered_map<std::string, std::vector<std::tuple<uint32_t,uint32_t>>> sent_queries;
     std::unordered_map<std::string, std::vector<std::string>> query_results; 
     std::atomic<bool> running;
     std::atomic<int> num_queries_to_send;

public:
     VortexPerfClient(int node_id, int num_queries, int batch_size, int query_interval, int emb_dim);

     int read_queries(std::filesystem::path query_filepath, std::vector<std::string>& queries);
     int read_query_embs(std::string query_emb_directory, std::unique_ptr<float[]>& query_embs);
     std::string format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list);

     /***
     * Result JSON is in format of : {"query": query_text, "top_k_docs":[doc_text1, doc_text2, ...], "query_batch_id": query_batch_id}
     */
     bool deserialize_result(const Blob& blob, std::string& query_text, std::vector<std::string>& top_k_docs,uint32_t& query_batch_id);
     
     /***
      * Register notification to all servers, helper function to run_perf_test
      * @return the number of shards that the connections are established
      *         -1 if failed to establish connections
      */
     int register_notification_on_all_servers(ServiceClientAPI& capi);

     /***
      * Run perf test
      */
     bool run_perf_test(ServiceClientAPI& capi, std::string& query_directory);

     bool flush_logs(ServiceClientAPI& capi, int num_shards);

     std::vector<std::vector<std::string>> read_groundtruth(std::filesystem::path filename);

     bool compute_recall(ServiceClientAPI& capi, std::string& query_directory);
};