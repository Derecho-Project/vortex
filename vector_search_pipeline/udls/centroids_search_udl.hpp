#include <memory>
#include <map>
#include <iostream>
#include <unordered_map>

#include "grouped_embeddings_for_search.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "10a2c111-1100-1100-1000-0001ac110000"
#define MY_DESC     "UDL search among the centroids to find the top num_centroids that the queries close to."
#define NEXT_UDL_SUBGROUP_ID 1 // cluster_search udl subgroup_id

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class CentroidsSearchOCDPO: public DefaultOffCriticalDataPathObserver {
    /* 
     * This is a thread for searching centroids. We can use multiple of this to search in parallel
     *
     */
    class CentroidSearchThread {
        private:
            uint64_t my_thread_id;
            CentroidsSearchOCDPO* parent;
            std::thread real_thread;
            bool running = false;

            std::condition_variable_any query_queue_cv;
            std::mutex query_queue_mutex;
            std::queue<std::shared_ptr<EmbeddingQuery>> query_queue;
            
            void main_loop();
            std::unique_ptr<std::vector<uint64_t>> centroid_search(std::shared_ptr<EmbeddingQuery> &query);

        public:
            CentroidSearchThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl);
            void start();
            void join();
            void signal_stop();

            void push(std::shared_ptr<EmbeddingQuery> query);
    };
    
    uint64_t num_search_threads = 1;
    uint64_t next_search_thread = 0;
    std::vector<std::unique_ptr<CentroidSearchThread>> search_threads;

    /***
     * This thread gathers queries in each cluster, batch them and emit to the next UDL
     */
    class BatchingThread {
        private:
            uint64_t my_thread_id;
            CentroidsSearchOCDPO* parent;
            std::thread real_thread;
            bool running = false;

            std::unordered_map<uint64_t,std::unique_ptr<std::vector<std::shared_ptr<EmbeddingQuery>>>> cluster_queue; // a queue for each cluster
            std::condition_variable_any cluster_queue_cv;
            std::mutex cluster_queue_mutex;

            void main_loop(DefaultCascadeContextType* typed_ctxt);

        public:
            BatchingThread(uint64_t thread_id, CentroidsSearchOCDPO* parent_udl);
            void start(DefaultCascadeContextType* typed_ctxt);
            void join();
            void signal_stop();
            void push(std::shared_ptr<EmbeddingQuery> query,std::unique_ptr<std::vector<uint64_t>> clusters);
    };

    std::unique_ptr<GroupedEmbeddingsForSearch> centroids_embs;
    bool cached_centroids_embs = false;
    int my_id = -1; // id of this node; logging purpose

    // values set by config in dfgs.json.tmp file
    std::string centroids_emb_prefix = "/rag/emb/centroids_obj";
    int emb_dim = 64; // dimension of each embedding
    int top_num_centroids = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search
    std::string emit_key_prefix = "/rag/emb/clusters_search";
    uint64_t min_batch_size = 1;
    uint64_t max_batch_size = 10;
    uint64_t batch_time_us = 1000;
    
    bool retrieve_and_cache_centroids_index(DefaultCascadeContextType* typed_ctxt);

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override;

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;

public:
    std::unique_ptr<BatchingThread> batch_thread;
    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<CentroidsSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config);

    void shutdown();
};

std::shared_ptr<OffCriticalDataPathObserver> CentroidsSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    CentroidsSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext* ctxt,const nlohmann::json& config) {
    auto typed_ctxt = dynamic_cast<DefaultCascadeContextType*>(ctxt);
    std::static_pointer_cast<CentroidsSearchOCDPO>(CentroidsSearchOCDPO::get())->set_config(typed_ctxt,config);
    return CentroidsSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    std::static_pointer_cast<CentroidsSearchOCDPO>(CentroidsSearchOCDPO::get())->shutdown();
    return;
}

} // namespace cascade
} // namespace derecho
