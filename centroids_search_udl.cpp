#include <cascade/user_defined_logic_interface.hpp>
#include <iostream>

#include "grouped_embeddings.hpp"

namespace derecho{
namespace cascade{

#define MY_UUID     "11a2c123-1100-21ac-1755-0001ac110000"
#define MY_DESC     "UDL search which centroids the queries close to."

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

class CentroidsSearchOCDPO: public DefaultOffCriticalDataPathObserver {

   
    // embeddings of centroids, assume that the embeddings are in order of the centroids from 0 to last_centroid_id
    std::unique_ptr<GroupedEmbeddings> centroid_embs;
    bool is_centroids_cached = false;
    /*** TODO: get this from dfgs config ***/
    // faiss example uses 64
    int emb_dim = 64; // dimension of each embedding
    /*** TODO: this is a duplicated field to num_embs of GroupedEmbeddings ***/
    // same as faiss example
    int num_embs = 100000; // number of embeddings

    /***
     * Fill in the memory cache by getting the centroids embeddings from KV store in Cascade
     * This function is called when this UDL is first triggered by caller to operator(),
     * the embeddings for all centroids are used to compute centroids search and find the closest clusters.
     * it sacrifices the first request to this node, but the following requests will benefit from this cache.
     * The embeddings for centroids are stored as KV objects in Cascade.
     * In static RAG setting, this function should be called only once at the begining.
     * In dynamic RAG setting, this function could be extended to call periodically or upon notification. 
     * (The reason not to call it at initialization, is that initialization is called upon server starts, 
     *  but the data have not been put to the servers yet, this needs to be called after the centroids data are put)
    ***/
    void fill_in_cached_centroids(){
        /*** TODO: implement this! ***/
        // 1. get the centroids from KV store in Cascade
        float* xb = new float[this->emb_dim * this->num_embs]; // Placeholder embeddings
        // 2. fill in the memory cache
        this->centroid_embs = std::make_unique<GroupedEmbeddings>(this->emb_dim, this->num_embs, xb, 1, 10000);
        // 3. set the flag to true
        this->is_centroids_cached = true;
        return;
    }

    virtual void ocdpo_handler(const node_id_t sender,
                               const std::string& object_pool_pathname,
                               const std::string& key_string,
                               const ObjectWithStringKey& object,
                               const emit_func_t& emit,
                               DefaultCascadeContextType* typed_ctxt,
                               uint32_t worker_id) override {
            std::cout << "[centroids search ocdpo_handler]: I(" << worker_id << ") received an object from sender:" << sender << " with key=" << key_string 
                << ", object_pool_pathname=" << object_pool_pathname << std::endl;
            // 0. check if local centroids cache is filled
            if (!this->is_centroids_cached) {
                fill_in_cached_centroids();
            }
            // 1. compute knn
            int nq = 10000;
            float* xq = new float[this->emb_dim * nq]; // Placeholder query embeddings
            this->centroid_embs->faiss_gpu_search(nq, xq);
            // 2. emit the result to the subsequent UDL
            Blob blob(object.blob);
            emit(key_string, EMIT_NO_VERSION_AND_TIMESTAMP , blob);
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
};

std::shared_ptr<OffCriticalDataPathObserver> CentroidsSearchOCDPO::ocdpo_ptr;

void initialize(ICascadeContext* ctxt) {
    CentroidsSearchOCDPO::initialize();
}

std::shared_ptr<OffCriticalDataPathObserver> get_observer(
        ICascadeContext*,const nlohmann::json&) {
    return CentroidsSearchOCDPO::get();
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
