#include <condition_variable>
#include <memory>
#include <map>
#include <iostream>
#include <shared_mutex>
#include <unordered_map>

#include <cascade/user_defined_logic_interface.hpp>
#include <cascade/utils.hpp>
#include <cascade/cascade_interface.hpp>
#include <derecho/openssl/hash.hpp>

#include "grouped_embeddings_for_search.hpp"
#include "api_utils.hpp"

#define EMIT_AGGREGATE_PREFIX "/rag/generate/agg"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

#define CLUSTER_EMB_OBJECTPOOL_PREFIX "/rag/emb/clusters/cluster"

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

struct queryQueue{
    std::vector<std::string> query_list;
    std::vector<std::string> query_keys;
    float* query_embs;
    std::atomic<int> added_query_offset;
    int emb_dim;

    queryQueue(int emb_dim);
    ~queryQueue();
    bool add_query(std::string&& query, std::string&& key, float* emb, int emb_dim);
    bool add_batched_queries(std::vector<std::string>&& queries, const std::string& key, float* embs, int emb_dim, int num_queries);
    bool could_add_query_nums(uint32_t num_queries);
    int count_queries();
    void reset();
};


class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {
    

    /***
     * A thread class to process ANN search on the queued queries in round-robin order on the local clusters
     *   then emit the result to the next UDL
     */
    class ClusterSearchWorker {
        uint64_t my_thread_id;
        ClustersSearchOCDPO* parent;

        bool running = false;
        std::thread real_thread;
        int cluster_id = -1; // the cluster that this worker handles

        std::unique_ptr<queryQueue> query_buffer;
        std::unique_ptr<queryQueue> shadow_query_buffer;
        std::atomic<int> use_shadow_flag;  // if this is 1, then add to shadow_query_buffer, otherwise add to query_buffer
        std::condition_variable_any query_buffer_cv;
        std::mutex query_buffer_mutex;
        


        bool check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt);
        /***
        * Format the new_keys for the search results of the queries
        * it is formated as client{client_id}qb{querybatch_id}qc{client_batch_query_count}_cluster{cluster_id}_qid{hash(query)}
        * @param new_keys the new keys to be constructed
        * @param key_string the key string of the object
        * @param query_list a list of query strings
        ***/
        void construct_new_keys(std::vector<std::string>& new_keys,
                                                       const std::vector<std::string>& query_keys, 
                                                       const std::vector<std::string>& query_list);
        void run_cluster_search_and_emit(DefaultCascadeContextType* typed_ctxt,
                                        queryQueue* query_buffer);
        /*** Helper function to check if there are enough pending queries on
         *   the buffer that have been added by the push_thread
         *   If so, then flip the flag to have the push_thread to start a new buffer, 
         *   while this thread process the queued queries
         *   @param num the number of queries to check if enough pending
         *   @return true if the accumulated queries > min_batch_size
         */
        bool enough_pending_queries(int num);

    public:
        ClusterSearchWorker(uint64_t thread_id, 
                            ClustersSearchOCDPO* parent_udl);
        void main_loop(DefaultCascadeContextType* typed_ctxt);   
        void push_to_query_buffer(int cluster_id, const Blob& blob, const std::string& key);
        // void push_queries_to_cluster(int cluster_id, std::vector<std::string>& query_list, std::vector<std::string>& query_keys);     
        void start(DefaultCascadeContextType* typed_ctxt);
        void join();
        void signal_stop();
    };

    // These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    uint32_t top_k = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
    std::string dataset_name {""};
    int my_id; // the node id of this node; logging purpose
    uint32_t batch_time_us=1000; // the time interval to process the batch of queries
    uint32_t batch_min_size=0; // min number queries to process in each batch

    mutable std::shared_mutex cluster_search_index_mutex;
    mutable std::condition_variable_any cluster_search_index_cv;
    // maps from cluster ID -> embeddings of that cluster, 
    // use pointer to allow multithreading adding queries to different GroupedEmbeddingsForSearch objects
    std::shared_ptr<GroupedEmbeddingsForSearch> cluster_search_index;    

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

public:
    /*** TODO: have a thread_pool for process cluster search workers */
    std::unique_ptr<ClusterSearchWorker> cluster_search_thread;

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void start_threads(DefaultCascadeContextType* typed_ctxt);

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);

    void shutdown();
};

std::shared_ptr<OffCriticalDataPathObserver> ClustersSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    ClustersSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->set_config(typed_ctxt,config);
    return ClustersSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "Error during cudaDeviceSynchronize in release: " << cudaGetErrorString(sync_err) << std::endl;
    } 
    std::static_pointer_cast<ClustersSearchOCDPO>(ClustersSearchOCDPO::get())->shutdown();  
    return;
}

} // namespace cascade
} // namespace derecho
