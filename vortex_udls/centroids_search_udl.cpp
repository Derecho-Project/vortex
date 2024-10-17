#include <memory>
#include <map>
#include <iostream>
#include <unordered_map>

#include "grouped_embeddings_for_search.hpp"


namespace derecho{
namespace cascade{

#define MY_UUID     "10a2c111-1100-1100-1000-0001ac110000"
#define MY_DESC     "UDL search among the centroids to find the top num_centroids that the queries close to."


std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class CentroidsSearchOCDPO: public DefaultOffCriticalDataPathObserver {

    std::unique_ptr<GroupedEmbeddingsForSearch> centroids_embs;
    bool cached_centroids_embs = false;

    // values set by config in dfgs.json.tmp file
    std::string centroids_emb_prefix = "/rag/emb/centroids_obj";
    int emb_dim = 64; // dimension of each embedding
    int top_num_centroids = 4; // number of top K embeddings to search
    int faiss_search_type = 0; // 0: CPU flat search, 1: GPU flat search, 2: GPU IVF search

    int my_id = -1; // id of this node; logging purpose

    /***
     * Combine subsets of queries that is going to send to the same cluster
     *  A batching step that batches the results with the same cluster in their top_num_centroids search results
     * @param I the indices of the top_num_centroids that are close to the queries
     * @param nq the number of queries
     * @param cluster_ids_to_query_ids a map from cluster_id to the list of query_ids that are close to the cluster
    ***/
    inline void combine_common_clusters(const long* I, const int nq, std::map<long, std::vector<int>>& cluster_ids_to_query_ids){
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < top_num_centroids; j++) {
                long cluster_id = I[i * top_num_centroids + j];
                if (cluster_ids_to_query_ids.find(cluster_id) == cluster_ids_to_query_ids.end()) {
                    cluster_ids_to_query_ids[cluster_id] = std::vector<int>();
                }
                cluster_ids_to_query_ids[cluster_id].push_back(i);
            }
        }
    }


    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
        /*** Note: this object_pool_pathname is trigger pathname prefix: /rag/emb/centroids_search instead of /rag/emb, i.e. the objp name***/
        dbg_default_trace("[Centroids search ocdpo]: I({}) received an object from sender:{} with key={}", worker_id, sender, key_string);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        // Logging purpose for performance evaluation
        if (key_string == "flush_logs") {
            std::string log_file_name = "node" + std::to_string(my_id) + "_udls_timestamp.dat";
            TimestampLogger::flush(log_file_name);
            std::cout << "Flushed logs to " << log_file_name <<"."<< std::endl;
            return;
        }
        int client_id = -1;
        int query_batch_id = -1;
        bool usable_logging_key = parse_batch_id(key_string, client_id, query_batch_id); // Logging purpose
        if (!usable_logging_key)
            dbg_default_error("Failed to parse client_id and query_batch_id from key: {}, unable to track correctly.", key_string);
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_START,client_id,query_batch_id,this->my_id);
#endif
        // 0. check if local cache contains the centroids' embeddings
        if (cached_centroids_embs == false ) {
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_LOADING_START,this->my_id,0,0);
#endif
            //  Fill centroids embs and keep it in memory cache
            int filled_centroid_embs = this->centroids_embs->retrieve_grouped_embeddings(this->centroids_emb_prefix,typed_ctxt);
            if (filled_centroid_embs == -1) {
                dbg_default_error("Failed to fill the centroids embeddings in cache, at centroids_search_udl.");
                return;
            }
            cached_centroids_embs = true;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_LOADING_END,this->my_id,0,0);
#endif
        }

        // 1. get the query embeddings from the object
        float* data;
        uint32_t nq;
        std::vector<std::string> query_list;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CENTROIDS_SEARCH_DESERIALIZE_START,client_id,query_batch_id,this->my_id);
#endif
        try{
            deserialize_embeddings_and_quries_from_bytes(object.blob.bytes,object.blob.size,nq,this->emb_dim,data,query_list);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to deserialize the query embeddings and query texts from the object." << std::endl;
            dbg_default_error("{}, Failed to deserialize the query embeddings and query texts from the object.", __func__);
            return;
        }
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CENTROIDS_SEARCH_DESERIALIZE_END,client_id,query_batch_id,this->my_id);
#endif

        // 2. search the top_num_centroids that are close to the query
        long* I = new long[this->top_num_centroids * nq];
        float* D = new float[this->top_num_centroids * nq];
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START,client_id,query_batch_id,this->my_id);
#endif
        try{
            this->centroids_embs->search(nq, data, this->top_num_centroids, D, I);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to search the top_num_centroids for the queries." << std::endl;
            dbg_default_error("{}, Failed to search the top_num_centroids for the queries.", __func__);
            return;
        }

#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END,client_id,query_batch_id,this->my_id);
#endif

        /*** 3. emit the result to the subsequent UDL
              trigger the subsequent UDL by evict the queries to shards that contains its top cluster_embs 
              according to affinity set sharding policy
        ***/
        std::map<long, std::vector<int>> cluster_ids_to_query_ids = std::map<long, std::vector<int>>();
        combine_common_clusters(I, nq, cluster_ids_to_query_ids);
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END,client_id,query_batch_id,this->my_id);
#endif
        for (const auto& pair : cluster_ids_to_query_ids) {
            if (pair.first == -1) {
                dbg_default_error( "Error: [CentroidsSearchOCDPO] for key: {} a selected cluster among top {}, has cluster_id -1", key_string, this->top_num_centroids);
                continue;
            }
            std::string new_key = key_string + "_cluster" + std::to_string(pair.first);
            std::vector<int> query_ids = pair.second;

            // create an bytes object by concatenating: num_queries + float array of emebddings + list of query_text
            uint32_t num_queries = static_cast<uint32_t>(query_ids.size());
            std::string nq_bytes(4, '\0');
            nq_bytes[0] = (num_queries >> 24) & 0xFF;
            nq_bytes[1] = (num_queries >> 16) & 0xFF;
            nq_bytes[2] = (num_queries >> 8) & 0xFF;
            nq_bytes[3] = num_queries & 0xFF;
            float* query_embeddings = new float[this->emb_dim * num_queries];
            for (uint32_t i = 0; i < num_queries; i++) {
                int query_id = query_ids[i];
                for (int j = 0; j < this->emb_dim; j++) {
                    query_embeddings[i * this->emb_dim + j] = data[query_id * this->emb_dim + j];
                }
            }
            std::vector<std::string> query_texts;
            for (uint32_t i = 0; i < num_queries; i++) {
                query_texts.push_back(query_list[query_ids[i]]);
            }
            // serialize the query embeddings and query texts, formated as num_queries + query_embeddings + query_texts
            std::string query_emb_string = nq_bytes +
                                        std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * this->emb_dim * num_queries) +
                                        nlohmann::json(query_texts).dump();
            Blob blob(reinterpret_cast<const uint8_t*>(query_emb_string.c_str()), query_emb_string.size());
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START,client_id,query_batch_id,pair.first);
#endif
            emit(new_key, EMIT_NO_VERSION_AND_TIMESTAMP , blob);
            std::cout << "[Centr] Emitted key: " << new_key << std::endl;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
            TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END,client_id,query_batch_id,pair.first);
#endif
            dbg_default_trace("[Centroids search ocdpo]: Emitted key: {}",new_key);
        }
        delete[] I;
        delete[] D;
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
        TimestampLogger::log(LOG_CENTROIDS_EMBEDDINGS_UDL_END,client_id,query_batch_id,this->my_id);
#endif
        dbg_default_trace("[Centroids search ocdpo]: FINISHED knn search for key: {}", key_string);
    }

    static std::shared_ptr<OffCriticalDataPathObserver> ocdpo_ptr;
public:

    static void initialize() {
        if(!ocdpo_ptr) {
            ocdpo_ptr = std::make_shared<CentroidsSearchOCDPO>();
        }
    }
    static auto get() {
        return ocdpo_ptr;
    }

    void set_config(DefaultCascadeContextType* typed_ctxt, const nlohmann::json& config){
        this->my_id = typed_ctxt->get_service_client_ref().get_my_id();
        try{
            if (config.contains("centroids_emb_prefix")) {
                this->centroids_emb_prefix = config["centroids_emb_prefix"].get<std::string>();
            }
            if (config.contains("emb_dim")) {
                this->emb_dim = config["emb_dim"].get<int>();
            }
            if (config.contains("top_num_centroids")) {
                this->top_num_centroids = config["top_num_centroids"].get<int>();
            }
            if (config.contains("faiss_search_type")) {
                this->faiss_search_type = config["faiss_search_type"].get<int>();
            }
            this->centroids_embs = std::make_unique<GroupedEmbeddingsForSearch>(this->faiss_search_type, this->emb_dim);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to convert emb_dim or top_num_centroids from config" << std::endl;
            dbg_default_error("Failed to convert emb_dim or top_num_centroids from config, at centroids_search_udl.");
        }
    }
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
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
