#pragma once
#include <queue>
#include <vector>
#include <string>
#include <cascade/service_client_api.hpp>

/* 
 * EmbeddingQuery encapsulates a single embedding query that is part of a batch. 
 * Operations are performed on demand and directly from the buffer of the whole batch.
 *
 */

class EmbeddingQuery {
    std::shared_ptr<uint8_t> buffer;
    uint64_t buffer_size;

    uint64_t query_id;
    uint32_t node_id,text_position,text_size,embeddings_position,embeddings_size;

    std::shared_ptr<std::string> text;

public:
    EmbeddingQuery(std::shared_ptr<uint8_t> buffer,uint64_t buffer_size,uint64_t query_id,uint32_t metadata_position);
    std::shared_ptr<std::string> get_text();
    const float * get_embeddings_pointer();
    const uint8_t * get_text_pointer();
    uint32_t get_text_size();
    uint64_t get_id();
    uint32_t get_node();

    friend class ClusterSearchResult;
};

/* 
 * EmbeddingQueryBatchManager perform operations on the whole embedding query batch received from the client or UDL1.
 * Such operations include getting all EmbeddingQuery that are in the batch, or getting all the embeddings for processing all in batch.
 *
 */

class EmbeddingQueryBatchManager {
    std::shared_ptr<uint8_t> buffer;
    uint64_t buffer_size;

    uint64_t emb_dim;
    uint32_t num_queries;
    uint32_t embeddings_position;
    bool copy_embeddings = true;

    uint32_t header_size;
    uint32_t metadata_size;
    uint32_t embeddings_size;

    std::vector<std::shared_ptr<EmbeddingQuery>> queries;
    
    void create_queries();

public:
    EmbeddingQueryBatchManager(const uint8_t *buffer,uint64_t buffer_size,uint64_t emb_dim,bool copy_embeddings = true);
    const std::vector<std::shared_ptr<EmbeddingQuery>>& get_queries();
    uint64_t count();
    uint32_t get_embeddings_position(uint32_t start = 0); // get the position of the embeddings in the buffer, starting at the query at position start
    uint32_t get_embeddings_size(uint32_t num = 0); // get the size of the embeddings buffer for the given number of queries
};

/* 
 * EmbeddingQueryBatcher gathers and serializes queries with their embeddings to be sent to UDL1 and to UDL2.
 *
 */

using query_id_t = uint64_t;
using queued_query_t = std::tuple<query_id_t,uint32_t,std::shared_ptr<float>,std::shared_ptr<std::string>>; // query ID, client node ID, embeddings, query text

class EmbeddingQueryBatcher {
    uint64_t emb_dim;
    uint32_t metadata_size;
    uint32_t header_size;
    uint32_t query_emb_size;
    uint32_t num_queries = 0;
    uint32_t total_text_size = 0;
    bool from_buffered = false;
    std::unordered_map<query_id_t,uint32_t> text_size;

    std::vector<queued_query_t> queries;
    std::vector<std::shared_ptr<EmbeddingQuery>> buffered_queries;
    std::shared_ptr<derecho::cascade::Blob> blob;
    
    void serialize_from_buffered();
    void serialize_from_raw();

public:
    EmbeddingQueryBatcher(uint64_t emb_dim,uint64_t size_hint = 1000);

    void add_query(queued_query_t &queued_query);
    void add_query(query_id_t query_id,uint32_t node_id,std::shared_ptr<float> query_emb,std::shared_ptr<std::string> query_text);
    void add_query(std::shared_ptr<EmbeddingQuery> query); 

    std::shared_ptr<derecho::cascade::Blob> get_blob();

    void serialize();
    void reset();
};

// in the format <query_id, client_id, query_text>
using encoder_query_t = std::tuple<query_id_t, uint32_t, std::shared_ptr<std::string>>;

/** Similar to EmbeddingQueryBatcher, but is used to gather and serialize queries
 * to the encoder UDL to be transformed into an EmbeddingQueryBatcher
 */
class EncoderQueryBatcher {
private:
    std::vector<encoder_query_t> _queries;

private:
    static constexpr uint32_t HEADER_SIZE = sizeof(uint32_t);
    static constexpr uint32_t METADATA_SIZE = sizeof(uint32_t) * 2 + sizeof(query_id_t);
    
    // maps from query_id to number of bytes used to encode query string
    std::unordered_map<query_id_t, uint32_t> _text_size_mapping;

    // blob representation of this class, updated when serialize() is called
    std::shared_ptr<derecho::cascade::Blob> _blob_repr;

public:
    // size_hint is used to reserve space in the underlying
    // vector for performance optimization
    EncoderQueryBatcher(uint64_t size_hint = 1000);

    void add_query(const encoder_query_t &query);
    void add_query(query_id_t query_id, uint32_t node_id, std::shared_ptr<std::string> query_text);

    std::shared_ptr<derecho::cascade::Blob> get_blob() const;

    void serialize();
    void reset();
};

/*
 * This encapsulates the results of the cluster search.
 * 
 */
class ClusterSearchResult {
    std::shared_ptr<long> ids;
    std::shared_ptr<float> dist;
    std::shared_ptr<uint8_t> buffer;

    uint64_t query_id;
    uint32_t top_k;
    uint32_t client_id,text_position,text_size,ids_position,ids_size,dist_position,dist_size;
    uint64_t cluster_id;

    std::shared_ptr<std::string> text;
    bool from_buffer = false;

public:
    ClusterSearchResult(std::shared_ptr<EmbeddingQuery> query,std::shared_ptr<long> ids,std::shared_ptr<float> dist,uint64_t idx,uint32_t top_k,uint64_t cluster_id); // created from an EmbeddingQuery and the results from FAISS (at UDL2)
    ClusterSearchResult(std::shared_ptr<uint8_t> buffer,uint64_t query_id,uint32_t metadata_position,uint32_t top_k,uint64_t cluster_id); // created from a serialized buffer (at UDL3)

    query_id_t get_query_id();
    uint32_t get_client_id();
    std::shared_ptr<std::string> get_text();
    const long * get_ids_pointer();
    uint32_t get_top_k();
    const float * get_distances_pointer();
    const uint8_t * get_text_pointer();
    uint32_t get_text_size();
    uint64_t get_cluster_id();
};

/*
 * This class gathers and serializes cluster search results to be sent to UDL3.
 *
 */
class ClusterSearchResultBatcher {
    uint32_t top_k;
    uint32_t metadata_size;
    uint32_t header_size;
    uint32_t ids_size;
    uint32_t dist_size;
    uint32_t num_results = 0;
    uint32_t total_text_size = 0;
    std::unordered_map<query_id_t,uint32_t> text_size;

    std::vector<std::shared_ptr<ClusterSearchResult>> results;
    std::shared_ptr<derecho::cascade::Blob> blob;

public:
    ClusterSearchResultBatcher(uint32_t top_k,uint64_t size_hint = 1000);

    void add_result(std::shared_ptr<ClusterSearchResult> result);

    std::shared_ptr<derecho::cascade::Blob> get_blob();

    void serialize();
    void reset();
};

/* 
 * This class manages a batch of cluster search results received from UDL2.
 *
 */

class ClusterSearchResultBatchManager {
    std::shared_ptr<uint8_t> buffer;
    uint64_t buffer_size;
    uint32_t num_results;
    uint32_t top_k;
    uint32_t header_size;
    uint32_t metadata_size;
    uint64_t cluster_id;

    std::vector<std::shared_ptr<ClusterSearchResult>> results;
    
    void create_results();

public:
    ClusterSearchResultBatchManager(const uint8_t *buffer,uint64_t buffer_size,uint64_t cluster_id);
    const std::vector<std::shared_ptr<ClusterSearchResult>>& get_results();
    uint64_t count();
};

class ClusterSearchResultsAggregate; // cross-reference

/*
 * Class for comparing document IDs based on their distance while adding elements to the priority queue
 */
class DocIDComparison {
    ClusterSearchResultsAggregate* aggregate;
public:
    DocIDComparison(ClusterSearchResultsAggregate* aggregate);
    DocIDComparison(const DocIDComparison &other);
    bool operator() (const long& l, const long& r) const;
};

/*
 * This is just a priority queue with an added method to access the underlying vector.
 * This way we can directly copy the top_k document IDs without removing one by one from the priority queue (we don't care about the order, just that we have the top k)
 *
 */
class AggregatePriorityQueue : public std::priority_queue<long,std::vector<long>,DocIDComparison> {
public:
    AggregatePriorityQueue(DocIDComparison &comp): std::priority_queue<long,std::vector<long>,DocIDComparison>(comp) {}

    /*
     * This is a bit hacky but does the job. 
     * Alternatively, we can just use a plain vector an manage it ourselves with std::push_heap and pop_heap
     */
    const std::vector<long>& get_vector() { return this->c; } 
};

class ClusterSearchResultsAggregate {
    uint32_t total_num_results;
    uint32_t received_results = 0;
    uint32_t top_k;
    std::shared_ptr<ClusterSearchResult> first_result; // so we can get data like query_text without copying
    std::unique_ptr<AggregatePriorityQueue> agg_top_k_results;
    std::unordered_map<long,float> distance;
    const std::unordered_map<uint64_t,std::unordered_map<long,long>> *cluster_doc_table;

public:
    ClusterSearchResultsAggregate(std::shared_ptr<ClusterSearchResult> result,uint32_t total_num_results, uint32_t top_k,const std::unordered_map<uint64_t,std::unordered_map<long,long>> *cluster_doc_table);

    void add_result(std::shared_ptr<ClusterSearchResult> result);
    bool all_results_received();

    query_id_t get_query_id();
    uint32_t get_client_id();
    const uint8_t * get_text_pointer();
    uint32_t get_text_size();
    std::shared_ptr<std::string> get_text();
    const std::vector<long>& get_ids();
    float get_distance(long id);
};

/*
 * This class gathers and serializes final results to send to the client.
 *
 */
class ClientNotificationBatcher {
    uint32_t top_k;
    uint32_t header_size;
    uint32_t query_ids_size;
    uint32_t doc_ids_size;
    uint32_t dist_size;
    uint32_t num_aggregates = 0;
    bool include_distances = false;

    std::vector<std::unique_ptr<ClusterSearchResultsAggregate>> aggregates;
    std::shared_ptr<derecho::cascade::Blob> blob;

public:
    ClientNotificationBatcher(uint32_t top_k,uint64_t size_hint = 1000,bool include_distances=false);

    void add_aggregate(std::unique_ptr<ClusterSearchResultsAggregate> aggregate);

    std::shared_ptr<derecho::cascade::Blob> get_blob();

    void serialize();
    void reset();
};

/*
 * This encapsulates the results of the ANN search.
 * 
 */
class VortexANNResult {
    std::shared_ptr<uint8_t> buffer;

    uint64_t query_id;
    uint32_t top_k;
    uint32_t ids_position;
    uint32_t dist_position;

public:
    VortexANNResult(std::shared_ptr<uint8_t> buffer,uint64_t query_id,uint32_t ids_position,uint32_t dist_position,uint32_t top_k);

    query_id_t get_query_id();
    const long * get_ids_pointer();
    uint32_t get_top_k();
    const float * get_distances_pointer();
};

/* 
 * This class manages a batch of client notifications received from UDL3.
 *
 */

class ClientNotificationManager {
    std::shared_ptr<uint8_t> buffer;
    uint64_t buffer_size;
    uint32_t num_results;
    uint32_t top_k;
    uint32_t header_size;
    uint32_t query_ids_size;
    uint32_t doc_ids_size;
    uint32_t dist_size;

    std::vector<std::shared_ptr<VortexANNResult>> results;
    
    void create_results();

public:
    ClientNotificationManager(std::shared_ptr<uint8_t> buffer,uint64_t buffer_size);
    const std::vector<std::shared_ptr<VortexANNResult>>& get_results();
    uint64_t count();
};


/*
 * Helper functions
 *
 */

std::pair<uint32_t,uint64_t> parse_client_and_batch_id(const std::string &str); // at UDL1
uint64_t parse_cluster_id(const std::string &str); // at UDL2
uint64_t parse_cluster_id_udl3(const std::string &str); // at UDL3

struct CompareObjKey {
     bool operator()(const std::string& key1, const std::string& key2) const {
          size_t pos = key1.rfind("/");
          int num1 = std::stoi(key1.substr(pos + 1));
          pos = key2.rfind("/");
          int num2 = std::stoi(key2.substr(pos + 1));
          return num1 > num2;
     }
};

/*** Helper function to callers of list_key:
*    filter keys that doesn't have exact prefix, or duplicate keys (from experiment at scale, it occurs.)
*    e.g. /doc1/1, /doc12/1, which return by list_keys("/doc1"), but are not for the same cluster
*    TODO: adjust op_list_keys semantics? 
*/
std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filter_exact_matched_keys(std::vector<std::string>& obj_keys, const std::string& prefix);

