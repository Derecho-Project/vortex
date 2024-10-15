#include <iostream>
#include <thread>

#include "search_worker.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-2200-21ac-1755-0002ac220000"
#define MY_DESC     "UDL search within the clusters to find the top K embeddings that the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class ClustersSearchOCDPO: public DefaultOffCriticalDataPathObserver {
    // These two values could be set by config in dfgs.json.tmp file
    int emb_dim = 64; // dimension of each embedding
    uint32_t top_k = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search

    // maps from cluster ID -> embeddings of that cluster, 
    // use std::unique_ptr to allow multithreading adding queries to different GroupedEmbeddingsForSearch objects
    std::unordered_map<int, std::unique_ptr<GroupedEmbeddingsForSearch>> cluster_search_index;

    int my_id; // the node id of this node; logging purpose

    mutable std::shared_mutex cluster_search_index_map_mutex;
    mutable std::condition_variable_any cluster_search_index_cv;
    std::atomic<bool> execution_thread_running = true;
    std::thread search_worker_thread;
    
private:
    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {

        /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/clusteres_search instead of /rag/emb, i.e. the objp name***/
        dbg_default_trace("[Clusters search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
        // 0. get the cluster ID
        int cluster_id;
        bool extracted_clusterid = parse_number(key_string, CLUSTER_KEY_DELIMITER, cluster_id); 
        if (!extracted_clusterid) {
            std::cerr << "Error: cluster ID not found in the key_string" << std::endl;
            dbg_default_error("Failed to find cluster ID from key: {}, at clusters_search_udl.", key_string);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        int client_id = -1;
        int query_batch_id = -1;
        bool usable_logging_key = parse_batch_id(key_string, client_id, query_batch_id); // Logging purpose
        if (!usable_logging_key)
            dbg_default_error("Failed to parse client_id and query_batch_id from key: {}, unable to track correctly.", key_string);
        TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_START,client_id,query_batch_id,cluster_id);
#endif
        // 1. check if local cache contains the embeddings of the cluster
        // std::unique_lock<std::mutex> lock(this->cluster_search_index_map_mutex);
        {
            std::shared_lock<std::shared_mutex> read_lock(cluster_search_index_map_mutex);
            auto it = this->cluster_search_index.find(cluster_id);
            if (it == this->cluster_search_index.end()){
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START,this->my_id,cluster_id,0);
#endif
                read_lock.unlock();
                std::unique_lock<std::shared_mutex> write_lock(cluster_search_index_map_mutex);
                // load the embeddings of the cluster from the cascade
                
                this->cluster_search_index[cluster_id]= std::make_unique<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
                std::string cluster_prefix = "/rag/emb/cluster" + std::to_string(cluster_id);
                int filled_cluster_embs = this->cluster_search_index[cluster_id]->retrieve_grouped_embeddings(cluster_prefix,typed_ctxt);
                if (filled_cluster_embs == -1) {
                    std::cerr << "Error: failed to fill the cluster embeddings in cache" << std::endl;
                    dbg_default_error("Failed to fill the cluster embeddings in cache, at clusters_search_udl.");
                    return;
                }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                TimestampLogger::log(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END,this->my_id,cluster_id,0);
#endif
                it = this->cluster_search_index.find(cluster_id);
            }
        }

        // 2. get the query embeddings from the object

        float* data;
        uint32_t nq;
        std::vector<std::string> query_list;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_DESERIALIZE_START,client_id,query_batch_id,cluster_id);
#endif
        try{
            deserialize_embeddings_and_quries_from_bytes(object.blob.bytes,object.blob.size,nq,this->emb_dim,data,query_list);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_DESERIALIZE_END,client_id,query_batch_id,cluster_id);
#endif
        cluster_search_index.at(cluster_id)->add_queries(nq, data, std::move(query_list), key_string);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CLUSTER_SEARCH_ADDED_TOBATCH,client_id,query_batch_id,cluster_id);
#endif
        cluster_search_index_cv.notify_one();
        dbg_default_trace("[Cluster search ocdpo]: FINISHED knn search for key: {}.", key_string );
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<ClustersSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
        this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
        try{
            if (config.contains("emb_dim")) {
                this->emb_dim = config["emb_dim"].get<int>();
            }
            if (config.contains("top_k")) {
                this->top_k = config["top_k"].get<uint32_t>();
            }
            if (config.contains("faiss_search_type")) {
                this->faiss_search_type = config["faiss_search_type"].get<int>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert emb_dim or top_k from config" << std::endl;
            dbg_default_error("Failed to convert emb_dim or top_k from config, at clusters_search_udl.");
        }
        search_worker_thread = std::thread([this, typed_ctxt]() {
        ClusterSearchWorker worker(static_cast<int>(top_k), cluster_search_index, cluster_search_index_cv,
                                       cluster_search_index_map_mutex, execution_thread_running);
            worker.search_and_emit(typed_ctxt);
        });
    }

    /*** TODO: double check the correct way to clean up thread */
    ~ClustersSearchOCDPO() {
        execution_thread_running = false;
        cluster_search_index_cv.notify_all();
        if (search_worker_thread.joinable()) {
            search_worker_thread.join();
        }
    }
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
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
