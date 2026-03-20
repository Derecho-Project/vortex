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


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

#define CLUSTER_EMB_OBJECTPOOL_PREFIX "/rag/emb/clusters/cluster"
#define EMIT_AGGREGATE_PREFIX "/rag/generate/agg"
#define AGGREGATE_SUBGROUP_INDEX 2

#define INITIAL_PENDING_BATCHES 10

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {
    /***
     * A thread class to process ANN search on batches of queries in round-robin order. 
     * Results are pushed to the batching thread queue to be sent to the next UDL.
     */
    class ClusterSearchWorker {
        uint64_t my_thread_id;
        ClustersSearchOCDPO* parent;

        bool running = false;
        std::thread real_thread;

        std::mutex query_buffer_mutex;
        std::condition_variable_any query_buffer_cv;
        std::vector<std::shared_ptr<PendingEmbeddingQueryBatch>> pending_batches;
        int64_t current_batch = -1;
        int64_t next_batch = 0;
        int64_t next_to_process = 0;

        std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> process_batch(std::shared_ptr<PendingEmbeddingQueryBatch> batch);

    public:
        ClusterSearchWorker(uint64_t thread_id, 
                            ClustersSearchOCDPO* parent_udl);
        void main_loop(DefaultCascadeContextType* typed_ctxt);   
        void push_queries(uint64_t cluster_id, std::unique_ptr<EmbeddingQueryBatchManager> batch_manager, const uint8_t *buffer);
        void start(DefaultCascadeContextType* typed_ctxt);
        void join();
        void signal_stop();
    };

    /***
     * This thread gathers queries for each shard, batch them and emit to the next UDL
     */
    class BatchingThread {
        private:
            uint64_t my_thread_id;
            ClustersSearchOCDPO* parent;
            ServiceClientAPI& capi = ServiceClientAPI::get_service_client();
            std::thread real_thread;
            bool running = false;

            std::unordered_map<uint32_t,std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>>> shard_queue; // a queue for each next shard
            std::condition_variable_any shard_queue_cv;
            std::mutex shard_queue_mutex;

            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            BatchingThread(uint64_t thread_id, ClustersSearchOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
            void push_results(std::unique_ptr<std::vector<std::shared_ptr<ClusterSearchResult>>> results);
    };

    // These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    uint32_t top_k = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
    int my_id; // the node id of this node; logging purpose
    uint32_t min_batch_size = 1; // min number of queries to send down the pipeline
    uint32_t max_batch_size = 100; // max number of queries to send down the pipeline
    uint32_t batch_time_us = 1000; // the time to wait for the minimum batch size
    uint32_t max_process_batch_size = 10; // max number of queries to process in each batch
                                          // for HNSW or Faiss CPU search, set it to a small value (e.g. 1 to 20): the goal is to better distribute the queries across threads
                                          // for GPU-based search, set it to a big value (e.g. 100s to 1000s): the goal is to better utilize GPU memory and cores

    int num_threads = 1; // number of threads to process the cluster search
    uint64_t next_thread = 0;

    int cluster_id = -1;
    std::shared_ptr<GroupedEmbeddingsForSearch> cluster_search_index;    

    bool check_and_retrieve_cluster_index(DefaultCascadeContextType* typed_ctxt);
    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

public:
    std::vector<std::unique_ptr<ClusterSearchWorker>> cluster_search_threads;
    std::unique_ptr<BatchingThread> batch_thread;

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
