#include <condition_variable>
#include <memory>
#include <shared_mutex>
#include <unordered_map>


#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>
#include <derecho/openssl/hash.hpp>

#include "grouped_embeddings_for_search.hpp"

#define EMIT_AGGREGATE_PREFIX "/rag/generate/agg"

namespace derecho{
namespace cascade{
class ClusterSearchWorker {
    int top_k;
    std::unordered_map<int, std::unique_ptr<GroupedEmbeddingsForSearch>>& cluster_search_index;
    std::condition_variable_any& cluster_search_index_cv;
    std::shared_mutex& cluster_search_index_map_mutex;
    // Keeps track of the last processed cluster for round-robin scheduling
    int last_processed_cluster_index = 0;
    std::atomic<bool>& execution_thread_running;

public:
    ClusterSearchWorker(int top_k,
                        std::unordered_map<int, std::unique_ptr<GroupedEmbeddingsForSearch>>& index,
                        std::condition_variable_any& cv,
                        std::shared_mutex& mutex,
                        std::atomic<bool>& running_flag)
        : top_k(top_k),
          cluster_search_index(index),
          cluster_search_index_cv(cv),
          cluster_search_index_map_mutex(mutex),
          execution_thread_running(running_flag) {}


    /***
     * Format the new_keys for the search results of the queries
     * it is formated as client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}_qid{hash(query)}
     * @param new_keys the new keys to be constructed
     * @param key_string the key string of the object
     * @param query_list a list of query strings
    ***/
    inline void construct_new_keys(std::vector<std::string>& new_keys,
                                                       const std::vector<std::string>& query_keys, 
                                                       const std::vector<std::string>& query_list) {
        for (size_t i = 0; i < query_keys.size(); ++i) {
            const std::string& key_string = query_keys[i];
            const std::string& query = query_list[i];
            std::string hashed_query;
            try {
                /*** TODO: do we need 32 bytes of hashed key? will simply int be sufficient? */
                uint8_t digest[32];
                openssl::Hasher sha256(openssl::DigestAlgorithm::SHA256);
                const char* query_cstr = query.c_str();
                sha256.hash_bytes(query_cstr, strlen(query_cstr), digest);
                std::ostringstream oss;
                for (int i = 0; i < 32; ++i) {
                    // Output each byte as a decimal value (0-255) without any leading zeros
                    oss << std::dec << static_cast<int>(digest[i]);
                }
                hashed_query = oss.str();
            } catch(openssl::openssl_error& ex) {
                dbg_default_error("Unable to compute SHA256 of typename. string = {}, exception name = {}", query, ex.what());
                throw;
            }
            std::string new_key = key_string + "_qid" + hashed_query;
            new_keys.push_back(new_key);
        }
    }


    void search_and_emit(DefaultCascadeContextType* typed_ctxt) {
        while (execution_thread_running) {
            std::unique_lock<std::shared_mutex> map_lock(cluster_search_index_map_mutex);

            cluster_search_index_cv.wait(map_lock, [this]() {
                for (const auto& [id, cluster_index] : cluster_search_index) {
                    if (cluster_index->has_pending_queries()) {
                        return true;
                    }
                }
                return !execution_thread_running;
            });

            if (!execution_thread_running) break;

            int clusters_size = cluster_search_index.size();
            if (clusters_size == 0) continue;

            auto it = cluster_search_index.begin();
            std::advance(it, last_processed_cluster_index % clusters_size);

            for (int j = 0; j < clusters_size; ++j) {
                if (it == cluster_search_index.end()) it = cluster_search_index.begin();
                auto& [cluster_id, cluster_index] = *it;

                if (cluster_index->has_pending_queries()) {
                    long* I = nullptr; // searched result index, which should be allocated by the batched Search function
                    float* D = nullptr; // searched result distance
                    std::vector<std::string> query_list;
                    std::vector<std::string> query_keys;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                    std::vector<std::tuple<int, int>> query_batch_infos;
                    for (const auto& key : query_keys) {
                        int client_id = -1;
                        int query_batch_id = -1;
                        parse_batch_id(key, client_id, query_batch_id); // Logging purpose
                        TimestampLogger::log(LOG_CLUSTER_SEARCH_FAISS_SEARCH_START,client_id,query_batch_id,cluster_id);
                        query_batch_infos.emplace_back(client_id, query_batch_id);
                    }
#endif
                    bool search_success = cluster_index->batchedSearch(top_k, &D, &I, query_list, query_keys);
                    if (!search_success || !I || !D) {
                        dbg_default_error("Failed to batch search for cluster: {}", cluster_id);
                        continue;
                    }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                    for (const auto& batch_info: query_batch_infos) {
                        TimestampLogger::log(LOG_CLUSTER_SEARCH_FAISS_SEARCH_END,std::get<0>(batch_info),std::get<1>(batch_info),cluster_id);
                    }
#endif
                    std::vector<std::string> new_keys;
                    construct_new_keys(new_keys, query_keys, query_list);

#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                    for (const auto& batch_info: query_batch_infos) {
                        TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_EMIT_START,std::get<0>(batch_info),std::get<1>(batch_info),cluster_id);
                    }
#endif
                    for (size_t k = 0; k < query_list.size(); ++k) {
                        ObjectWithStringKey obj;
                        obj.key = std::string(EMIT_AGGREGATE_PREFIX) + "/" + new_keys[k];
                        std::string query_emit_content = serialize_cluster_search_result(top_k, I, D, k, query_list[k]);
                        obj.blob = Blob(reinterpret_cast<const uint8_t*>(query_emit_content.c_str()), query_emit_content.size());
                        typed_ctxt->get_service_client_ref().put_and_forget(obj);
                        std::cout << "Emitting key: " << obj.key << std::endl;
                    }

                    delete[] I;
                    delete[] D;

                    last_processed_cluster_index = (std::distance(cluster_search_index.begin(), it) + 1) % clusters_size;
                    ++it;
                    break;
                }
                ++it;
            }
        }
    }
};

} // namespace cascade
} // namespace derecho