#include <iostream>    
#include <limits>      
#include <stdexcept>   
#include <unordered_set>
#include <cascade/utils.hpp>
#include "serialize_utils.hpp"


/*
 * EmbeddingQueryBatcher implementation
 */

EmbeddingQueryBatcher::EmbeddingQueryBatcher(uint64_t emb_dim,uint64_t size_hint){
    this->emb_dim = emb_dim;
    metadata_size = sizeof(uint32_t) * 5 + sizeof(query_id_t);
    header_size = sizeof(uint32_t) * 2;
    query_emb_size = sizeof(float) * emb_dim;
    
    queries.reserve(size_hint);
    buffered_queries.reserve(size_hint);
}

void EmbeddingQueryBatcher::add_query(queued_query_t &queued_query){
    from_buffered = false;
    queries.emplace_back(queued_query);
}

void EmbeddingQueryBatcher::add_query(query_id_t query_id,uint32_t node_id,std::shared_ptr<float> query_emb,std::shared_ptr<std::string> query_text){
    queued_query_t queued_query(query_id,node_id,query_emb,query_text);
    add_query(queued_query);
}

void EmbeddingQueryBatcher::add_query(std::shared_ptr<EmbeddingQuery> query){
    from_buffered = true;
    buffered_queries.push_back(query);
}

void EmbeddingQueryBatcher::serialize(){
    if(from_buffered){
        serialize_from_buffered();
    } else {
        serialize_from_raw();
    }
}
    
void EmbeddingQueryBatcher::serialize_from_buffered(){
    num_queries = buffered_queries.size();
    total_text_size = 0;
    uint32_t total_size = header_size; // header: num_queries, embeddings_start_position

    // compute the number of bytes each query will take in the buffer
    for(auto& query : buffered_queries){
        query_id_t query_id = query->get_id();
        uint32_t query_text_size = query->get_text_size();
        total_size += query_text_size + metadata_size + query_emb_size;
        total_text_size += query_text_size;
        text_size[query_id] = query_text_size;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t metadata_position = header_size; // position to start writing metadata
            uint32_t text_position = metadata_position + (num_queries * metadata_size); // position to start writing the query texts
            uint32_t embeddings_position = text_position + total_text_size; // position to start writing the embeddings
            
            // write header
            uint32_t header[2] = {num_queries,embeddings_position};
            std::memcpy(buffer,header,header_size);
            
            // write each query to the buffer, starting at buffer_position
            for(auto& query : buffered_queries){
                query_id_t query_id = query->get_id();
                uint32_t node_id = query->get_node();
                const float* query_emb = query->get_embeddings_pointer();
                const uint8_t * text_data = query->get_text_pointer();
                uint32_t query_text_size = text_size[query_id];

                // write metadata: query_id, node_id, query_text_position, query_text_size, embeddings_position, query_emb_size
                uint32_t metadata_array[5] = {node_id,text_position,query_text_size,embeddings_position,query_emb_size};
                std::memcpy(buffer+metadata_position,&query_id,sizeof(query_id_t));
                std::memcpy(buffer+metadata_position+sizeof(query_id_t),metadata_array,metadata_size-sizeof(query_id_t));

                // write embeddings
                std::memcpy(buffer+embeddings_position,query_emb,query_emb_size);
                
                // write text
                std::memcpy(buffer+text_position,text_data,query_text_size);
               
                // update position for the next 
                metadata_position += metadata_size;
                embeddings_position += query_emb_size;
                text_position += query_text_size;
            }

            return size;
        },total_size);
}

void EmbeddingQueryBatcher::serialize_from_raw(){
    num_queries = queries.size();
    total_text_size = 0;
    uint32_t total_size = header_size; // header: num_queries, embeddings_start_position

    // compute the number of bytes each query will take in the buffer
    for(auto& queued_query : queries){
        query_id_t query_id = std::get<0>(queued_query);
        const std::string& query_txt = *std::get<3>(queued_query);

        uint32_t query_text_size = mutils::bytes_size(query_txt);
        total_text_size += query_text_size;
        total_size += query_text_size + metadata_size + query_emb_size;
        text_size[query_id] = query_text_size;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t metadata_position = header_size; // position to start writing metadata
            uint32_t text_position = metadata_position + (num_queries * metadata_size); // position to start writing the query texts
            uint32_t embeddings_position = text_position + total_text_size; // position to start writing the embeddings
             
            // write header
            uint32_t header[2] = {num_queries,embeddings_position};
            std::memcpy(buffer,header,header_size);

            // write each query to the buffer, starting at buffer_position
            for(auto& queued_query : queries){
                query_id_t query_id = std::get<0>(queued_query);
                uint32_t node_id = std::get<1>(queued_query);
                const float* query_emb = std::get<2>(queued_query).get();
                const std::string& query_txt = *std::get<3>(queued_query);
                uint32_t query_text_size = text_size[query_id];

                // write metadata: query_id, node_id, query_text_position, query_text_size, embeddings_position, query_emb_size
                uint32_t metadata_array[5] = {node_id,text_position,query_text_size,embeddings_position,query_emb_size};
                std::memcpy(buffer+metadata_position,&query_id,sizeof(query_id_t));
                std::memcpy(buffer+metadata_position+sizeof(query_id_t),metadata_array,metadata_size-sizeof(query_id_t));

                // write embeddings
                std::memcpy(buffer+embeddings_position,query_emb,query_emb_size);
                
                // write text
                mutils::to_bytes(query_txt,buffer+text_position);
               
                // update position for the next 
                metadata_position += metadata_size;
                embeddings_position += query_emb_size;
                text_position += query_text_size;
            }

            return size;
        },total_size);
}

std::shared_ptr<derecho::cascade::Blob> EmbeddingQueryBatcher::get_blob(){
    return blob;
}

void EmbeddingQueryBatcher::reset(){
    blob.reset();
    queries.clear();
    buffered_queries.clear();
    text_size.clear();
}

/*
 * EmbeddingQuery implementation
 */

EmbeddingQuery::EmbeddingQuery(std::shared_ptr<uint8_t> buffer,uint64_t buffer_size,uint64_t query_id,uint32_t metadata_position){
    this->buffer = buffer;
    this->buffer_size = buffer_size;
    this->query_id = query_id;

    // get metadata
    const uint32_t *metadata = reinterpret_cast<uint32_t*>(buffer.get()+metadata_position+sizeof(query_id_t));
    node_id = metadata[0];
    text_position = metadata[1];
    text_size = metadata[2];
    embeddings_position = metadata[3];
    embeddings_size = metadata[4];
}

std::shared_ptr<std::string> EmbeddingQuery::get_text(){
    if(!text){
        text = mutils::from_bytes<std::string>(nullptr,buffer.get()+text_position);
    }

    return text;
}

const float * EmbeddingQuery::get_embeddings_pointer(){
    if(embeddings_position >= buffer_size){
        return nullptr;
    }

    return reinterpret_cast<float*>(buffer.get()+embeddings_position);
}

const uint8_t * EmbeddingQuery::get_text_pointer(){
    return buffer.get()+text_position;
}

uint32_t EmbeddingQuery::get_text_size(){
    return text_size;
}

uint64_t EmbeddingQuery::get_id(){
    return query_id;
}

uint32_t EmbeddingQuery::get_node(){
    return node_id;
}


/*
 * EmbeddingQueryBatchManager implementation
 */

EmbeddingQueryBatchManager::EmbeddingQueryBatchManager(const uint8_t *buffer,uint64_t buffer_size,uint64_t emb_dim,bool copy_embeddings){
    this->emb_dim = emb_dim;
    this->copy_embeddings = copy_embeddings;
    
    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer);
    this->num_queries = header[0];
    this->embeddings_position = header[1];
    
    this->header_size = sizeof(uint32_t) * 2;
    this->metadata_size = sizeof(uint32_t) * 5 + sizeof(query_id_t);
    this->embeddings_size = buffer_size - this->embeddings_position;
   
    if(copy_embeddings){
        this->buffer_size = buffer_size;
    } else {
        this->buffer_size = buffer_size - this->embeddings_size;
    }

    std::shared_ptr<uint8_t> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(),buffer,this->buffer_size);
    this->buffer = std::move(copy);
}

const std::vector<std::shared_ptr<EmbeddingQuery>>& EmbeddingQueryBatchManager::get_queries(){
    if(queries.empty()){
        create_queries();
    }

    return queries;
}

uint64_t EmbeddingQueryBatchManager::count(){
    return num_queries;
}

uint32_t EmbeddingQueryBatchManager::get_embeddings_position(uint32_t start){
    return embeddings_position + (start * (emb_dim * sizeof(float)));
}

uint32_t EmbeddingQueryBatchManager::get_embeddings_size(uint32_t num){
    if(num == 0){
        return this->embeddings_size;
    }

    return num * emb_dim * sizeof(float);
}

void EmbeddingQueryBatchManager::create_queries(){
    for(uint32_t i=0;i<num_queries;i++){
        uint32_t metadata_position = header_size + (i * metadata_size);
        query_id_t query_id = *reinterpret_cast<query_id_t*>(buffer.get()+metadata_position);
        queries.emplace_back(new EmbeddingQuery(buffer,buffer_size,query_id,metadata_position));
    }
}

/*
 * ClusterSearchResult implementation
 */

ClusterSearchResult::ClusterSearchResult(std::shared_ptr<EmbeddingQuery> query,std::shared_ptr<long> ids,std::shared_ptr<float> dist,uint64_t idx,uint32_t top_k,uint64_t cluster_id){
    // from query
    query_id = query->query_id;
    client_id = query->node_id;
    text_position = query->text_position;
    text_size = query->text_size;
    buffer = query->buffer;

    this->ids = ids;
    this->dist = dist;
    this->top_k = top_k;
    this->cluster_id = cluster_id;

    ids_size = top_k * sizeof(long);
    ids_position = idx * top_k;
        
    dist_size = top_k * sizeof(float);
    dist_position = idx * top_k;
    
    from_buffer = false;
}

ClusterSearchResult::ClusterSearchResult(std::shared_ptr<uint8_t> buffer,uint64_t query_id,uint32_t metadata_position,uint32_t top_k,uint64_t cluster_id){
    this->buffer = buffer;
    this->query_id = query_id;
    this->top_k = top_k;
    this->cluster_id = cluster_id;

    // get metadata: client_id,text_position,text_size,ids_position,ids_size,dist_position,dist_size
    const uint32_t *metadata = reinterpret_cast<uint32_t*>(buffer.get()+metadata_position+sizeof(query_id_t));
    client_id = metadata[0];
    text_position = metadata[1];
    text_size = metadata[2];
    ids_position = metadata[3];
    ids_size = metadata[4];
    dist_position = metadata[5];
    dist_size = metadata[6];
    
    from_buffer = true;
}

uint32_t ClusterSearchResult::get_top_k(){
    return top_k;
}

uint64_t ClusterSearchResult::get_cluster_id(){
    return cluster_id;
}

std::shared_ptr<std::string> ClusterSearchResult::get_text(){
    if(!text){
        text = mutils::from_bytes<std::string>(nullptr,buffer.get()+text_position);
    }

    return text;
}

const long * ClusterSearchResult::get_ids_pointer(){
    if(from_buffer){
        return reinterpret_cast<long*>(buffer.get()+ids_position);
    }

    return ids.get() + ids_position;
}

const float * ClusterSearchResult::get_distances_pointer(){
    if(from_buffer){
        return reinterpret_cast<float*>(buffer.get()+dist_position);
    }

    return dist.get() + dist_position;
}

const uint8_t * ClusterSearchResult::get_text_pointer(){
    return buffer.get()+text_position;
}

uint32_t ClusterSearchResult::get_text_size(){
    return text_size;
}

query_id_t ClusterSearchResult::get_query_id(){
    return query_id;
}

uint32_t ClusterSearchResult::get_client_id(){
    return client_id;
}

/*
 * ClusterSearchResultBatcher implementation
 */

ClusterSearchResultBatcher::ClusterSearchResultBatcher(uint32_t top_k,uint64_t size_hint){
    this->top_k = top_k;

    results.reserve(size_hint);

    metadata_size = sizeof(uint32_t) * 7 + sizeof(query_id_t);
    header_size = sizeof(uint32_t) * 2;
    ids_size = top_k * sizeof(long);
    dist_size = top_k * sizeof(float);
}

void ClusterSearchResultBatcher::add_result(std::shared_ptr<ClusterSearchResult> result){
    results.push_back(result);
}

// format: num_results,top_k | {query_id,client_id,text_position,text_size,ids_position,ids_size,dist_position,dist_size} | {query_text} | {ids_array} | {distances_array}
void ClusterSearchResultBatcher::serialize(){
    num_results = results.size();
    total_text_size = 0;
    uint32_t total_size = header_size; // header: num_results,top_k

    // compute the number of bytes each result will take in the buffer
    for(auto& res : results){
        query_id_t query_id = res->get_query_id();
        uint32_t query_text_size = res->get_text_size();
        total_size += query_text_size + metadata_size + ids_size + dist_size;
        total_text_size += query_text_size;
        text_size[query_id] = query_text_size;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t metadata_position = header_size; // position to start writing metadata
            uint32_t text_position = metadata_position + (num_results * metadata_size); // position to start writing the query texts
            uint32_t ids_position = text_position + total_text_size; // position to start writing the IDs
            uint32_t dist_position = ids_position + (num_results * ids_size); // position to start writing the distances
            
            // write header
            uint32_t header[2] = {num_results,top_k};
            std::memcpy(buffer,header,header_size);
            
            // write each result to the buffer
            for(auto& res : results){
                query_id_t query_id = res->get_query_id();
                uint32_t node_id = res->get_client_id();
                const long * res_ids = res->get_ids_pointer();
                const float * res_dist = res->get_distances_pointer();
                const uint8_t * text_data = res->get_text_pointer();
                uint32_t res_text_size = text_size[query_id];

                // write metadata: query_id, node_id, text_position, res_text_size, ids_position, ids_size, dist_position, dist_size
                uint32_t metadata_array[7] = {node_id,text_position,res_text_size,ids_position,ids_size,dist_position,dist_size};
                std::memcpy(buffer+metadata_position,&query_id,sizeof(query_id_t));
                std::memcpy(buffer+metadata_position+sizeof(query_id_t),metadata_array,metadata_size-sizeof(query_id_t));

                // write ids
                std::memcpy(buffer+ids_position,res_ids,ids_size);
                
                // write dist
                std::memcpy(buffer+dist_position,res_dist,dist_size);
                
                // write text
                std::memcpy(buffer+text_position,text_data,res_text_size);
               
                // update position for the next 
                metadata_position += metadata_size;
                text_position += res_text_size;
                ids_position += ids_size;
                dist_position += dist_size;
            }

            return size;
        },total_size);
}

std::shared_ptr<derecho::cascade::Blob> ClusterSearchResultBatcher::get_blob(){
    return blob;
}

void ClusterSearchResultBatcher::reset(){
    blob.reset();
    results.clear();
    text_size.clear();
}

/*
 * ClusterSearchResultBatchManager implementation
 */

ClusterSearchResultBatchManager::ClusterSearchResultBatchManager(const uint8_t *buffer,uint64_t buffer_size,uint64_t cluster_id){
    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer);
    this->num_results = header[0];
    this->top_k = header[1];
    this->cluster_id = cluster_id;
    
    this->header_size = sizeof(uint32_t) * 2;
    this->metadata_size = sizeof(uint32_t) * 7 + sizeof(query_id_t);
    this->buffer_size = buffer_size;

    std::shared_ptr<uint8_t> copy(new uint8_t[this->buffer_size]);
    std::memcpy(copy.get(),buffer,this->buffer_size);
    this->buffer = std::move(copy);
}

const std::vector<std::shared_ptr<ClusterSearchResult>>& ClusterSearchResultBatchManager::get_results(){
    if(results.empty()){
        create_results();
    }

    return results;
}

uint64_t ClusterSearchResultBatchManager::count(){
    return num_results;
}

void ClusterSearchResultBatchManager::create_results(){
    results.reserve(num_results);

    for(uint32_t i=0;i<num_results;i++){
        uint32_t metadata_position = header_size + (i * metadata_size);
        query_id_t query_id = *reinterpret_cast<query_id_t*>(buffer.get()+metadata_position);
        results.emplace_back(new ClusterSearchResult(buffer,query_id,metadata_position,top_k,cluster_id));
    }
}

/*
 * ClusterSearchResultsAggregate implementation
 *
 */

DocIDComparison::DocIDComparison(ClusterSearchResultsAggregate* aggregate): aggregate(aggregate) {}

DocIDComparison::DocIDComparison(const DocIDComparison &other): aggregate(other.aggregate) {}

bool DocIDComparison::operator() (const long& l, const long& r) const {
    return aggregate->get_distance(l) < aggregate->get_distance(r);
}

ClusterSearchResultsAggregate::ClusterSearchResultsAggregate(std::shared_ptr<ClusterSearchResult> result,uint32_t total_num_results, uint32_t top_k,const std::unordered_map<uint64_t,std::unordered_map<long,long>> *cluster_doc_table) {
    this->total_num_results = total_num_results;
    this->received_results = 0;
    this->top_k = top_k;
    this->first_result = result;
    this->cluster_doc_table = cluster_doc_table;

    DocIDComparison comp(this);
    this->agg_top_k_results = std::make_unique<AggregatePriorityQueue>(comp);

    add_result(result);
}

bool ClusterSearchResultsAggregate::all_results_received(){
    return received_results >= total_num_results;
}

void ClusterSearchResultsAggregate::add_result(std::shared_ptr<ClusterSearchResult> result){
    // TODO should we check if this result has already been received? may be important in the future when/if we have fault tolerance

    // add the doc IDs to the max heap, and keep the size of the heap to be top_k
    const long * ids = result->get_ids_pointer();
    const float * dist = result->get_distances_pointer();
    uint64_t cluster_id = result->get_cluster_id();
    const auto& doc_table = cluster_doc_table->at(cluster_id);
    for(uint32_t i=0; i<result->get_top_k(); i++){
        long doc_id = doc_table.at(ids[i]); // map local id to global id
        distance[doc_id] = dist[i];

        if (agg_top_k_results->size() < top_k) {
            agg_top_k_results->push(doc_id);
        } else {
            long top_id = agg_top_k_results->top();
            if (distance[doc_id] < distance[top_id]) {
                agg_top_k_results->pop();
                agg_top_k_results->push(doc_id);
            }
        }
    }

    received_results++;
}


query_id_t ClusterSearchResultsAggregate::get_query_id(){
    return first_result->get_query_id();
}

uint32_t ClusterSearchResultsAggregate::get_client_id(){
    return first_result->get_client_id();
}

const uint8_t * ClusterSearchResultsAggregate::get_text_pointer(){
    return first_result->get_text_pointer();
}

uint32_t ClusterSearchResultsAggregate::get_text_size(){
    return first_result->get_text_size();
}

std::shared_ptr<std::string> ClusterSearchResultsAggregate::get_text(){
    return first_result->get_text();
}

const std::vector<long>& ClusterSearchResultsAggregate::get_ids(){
    return agg_top_k_results->get_vector();
}

float ClusterSearchResultsAggregate::get_distance(long id){
    return distance[id];
}

/*
 * ClientNotificationBatcher implementation
 */

ClientNotificationBatcher::ClientNotificationBatcher(uint32_t top_k,uint64_t size_hint,bool include_distances){
    this->top_k = top_k;
    this->include_distances = include_distances;

    aggregates.reserve(size_hint);

    header_size = sizeof(uint32_t) * 2;
    query_ids_size = sizeof(query_id_t);
    doc_ids_size = top_k * sizeof(long);
    dist_size = top_k * sizeof(float);
}

void ClientNotificationBatcher::add_aggregate(std::unique_ptr<ClusterSearchResultsAggregate> aggregate){
    aggregates.push_back(std::move(aggregate));
}

// format: num_aggregates,top_k | {query_id} | {doc_ids} | {dist}
void ClientNotificationBatcher::serialize(){
    num_aggregates = aggregates.size();
    uint32_t total_size = header_size + (query_ids_size * num_aggregates) + (doc_ids_size * num_aggregates); 
    if(include_distances){
        total_size += dist_size * num_aggregates;
    }

    // use a lambda to build buffer, to avoid a copy
    blob = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer,const std::size_t size){
            uint32_t query_ids_position = header_size;
            uint32_t doc_ids_position = query_ids_position + (query_ids_size * num_aggregates);
            uint32_t dist_position = doc_ids_position + (num_aggregates * doc_ids_size);
            
            // write header
            uint32_t header[2] = {num_aggregates,top_k};
            std::memcpy(buffer,header,header_size);
            
            // write each result to the buffer
            for(auto& agg : aggregates){
                query_id_t query_id = agg->get_query_id();
                const long * ids_data = agg->get_ids().data();
            
                // write query_id
                std::memcpy(buffer+query_ids_position,&query_id,query_ids_size);

                // write doc ids
                std::memcpy(buffer+doc_ids_position,ids_data,doc_ids_size);
                
                // write distances
                if(include_distances){
                    float *dist_buffer = reinterpret_cast<float*>(buffer+dist_position);
                    for(uint32_t i=0;i<top_k;i++){
                        float dist = agg->get_distance(ids_data[i]);
                        dist_buffer[i] = dist;
                    }
                }

                // update position for the next 
                query_ids_position += query_ids_size;
                doc_ids_position += doc_ids_size;
                dist_position += dist_size;
            }

            return size;
        },total_size);
}

std::shared_ptr<derecho::cascade::Blob> ClientNotificationBatcher::get_blob(){
    return blob;
}

void ClientNotificationBatcher::reset(){
    blob.reset();
    aggregates.clear();
}

/*
 * ClientNotificationManager implementation
 */

ClientNotificationManager::ClientNotificationManager(std::shared_ptr<uint8_t> buffer,uint64_t buffer_size){
    this->buffer = buffer;
    this->buffer_size = buffer_size;

    const uint32_t *header = reinterpret_cast<const uint32_t *>(buffer.get());
    this->num_results = header[0];
    this->top_k = header[1];
    
    this->header_size = sizeof(uint32_t) * 2;
    this->query_ids_size = sizeof(query_id_t);
    this->doc_ids_size = top_k * sizeof(long);
    this->dist_size = top_k * sizeof(float);
}

const std::vector<std::shared_ptr<VortexANNResult>>& ClientNotificationManager::get_results(){
    if(results.empty()){
        create_results();
    }

    return results;
}

uint64_t ClientNotificationManager::count(){
    return num_results;
}

void ClientNotificationManager::create_results(){
    results.reserve(num_results);

    uint32_t ids_start = header_size + (num_results * query_ids_size);
    uint32_t dist_start = ids_start + (num_results * doc_ids_size);

    for(uint32_t i=0;i<num_results;i++){
        // VortexANNResult(std::shared_ptr<uint8_t> buffer,uint64_t query_id,uint32_t ids_position,uint32_t dist_position,uint32_t top_k);
        uint32_t metadata_position = header_size + (i * query_ids_size);
        query_id_t query_id = *reinterpret_cast<query_id_t*>(buffer.get()+metadata_position);
        
        uint32_t ids_position = ids_start + (i * doc_ids_size);
        uint32_t dist_position = 0;
        if(dist_start < buffer_size){
            dist_position = dist_start + (i * dist_size);
        }

        results.emplace_back(new VortexANNResult(buffer,query_id,ids_position,dist_position,top_k));
    }
}

/*
 * VortexANNResult implementation
 */

VortexANNResult::VortexANNResult(std::shared_ptr<uint8_t> buffer,uint64_t query_id,uint32_t ids_position,uint32_t dist_position,uint32_t top_k){
    this->buffer = buffer;
    this->query_id = query_id;
    this->top_k = top_k;
    this->ids_position = ids_position;
    this->dist_position = dist_position;
}

uint32_t VortexANNResult::get_top_k(){
    return top_k;
}

const long * VortexANNResult::get_ids_pointer(){
    return reinterpret_cast<long*>(buffer.get()+ids_position);
}

const float * VortexANNResult::get_distances_pointer(){
    return reinterpret_cast<float*>(buffer.get()+dist_position);
}

query_id_t VortexANNResult::get_query_id(){
    return query_id;
}

/*
 * Helper functions
 *
 */

std::pair<uint32_t,uint64_t> parse_client_and_batch_id(const std::string &str){
    size_t pos = str.find("_");
    uint32_t client_id = std::stoll(str.substr(0,pos));
    uint64_t batch_id = std::stoull(str.substr(pos+1));
    return std::make_pair(client_id,batch_id);
}

uint64_t parse_cluster_id(const std::string &str){
    return std::stoull(str.substr(8)); // str is '/cluster[0-9]+'
}

uint64_t parse_cluster_id_udl3(const std::string &str){
    size_t pos = str.find("_cluster");
    return std::stoull(str.substr(pos+8)); // str is '/agg/results_cluster[0-9]+'
}

/*** Helper function to callers of list_key:
*    filter keys that doesn't have exact prefix, or duplicate keys (from experiment at scale, it occurs.)
*    e.g. /doc1/1, /doc12/1, which return by list_keys("/doc1"), but are not for the same cluster
*    TODO: adjust op_list_keys semantics? 
*/
std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filter_exact_matched_keys(std::vector<std::string>& obj_keys, const std::string& prefix){
     std::priority_queue<std::string, std::vector<std::string>, CompareObjKey> filtered_keys;
     std::unordered_set<std::string> key_set; /*** TODO: only for correctness test*/
     for (auto& key : obj_keys) {
          size_t pos = key.rfind("/");
          if (pos == std::string::npos) {
               std::cerr << "Error: invalid obj_key format, key=" << key << "prefix" << prefix  << std::endl; // shouldn't happen
               continue;
          }
          if (key.substr(0, pos) == prefix && key_set.find(key) == key_set.end()) {
               filtered_keys.push(key);
               key_set.insert(key);
          }
     }
     if (key_set.size() != filtered_keys.size()) {
          std::cerr << "Error: filter_exact_matched_keys: key_set.size()=" << key_set.size() << ",filtered_keys.size()=" << filtered_keys.size() << std::endl;
     }
     return filtered_keys;
}

/// ENCODER QUERY BATCHER IMPL
///
///
///

EncoderQueryBatcher::EncoderQueryBatcher(uint64_t size_hint) {
    _queries.reserve(size_hint);
} 

void EncoderQueryBatcher::add_query(const encoder_query_t &query) {
   _queries.emplace_back(query);
}

void EncoderQueryBatcher::add_query(query_id_t query_id, uint32_t node_id, std::shared_ptr<std::string> query_text) {
    encoder_query_t encoder_query {query_id, node_id, query_text};
    add_query(encoder_query);
}

std::shared_ptr<derecho::cascade::Blob> EncoderQueryBatcher::get_blob() const{
    return _blob_repr;
}

void EncoderQueryBatcher::reset() {
    _blob_repr.reset();
    _queries.clear();
}

void EncoderQueryBatcher::serialize() {
    // compute number of bytes to serialize all strings 
    uint32_t total_text_size_bytes = 0;
    uint32_t total_size_bytes = EncoderQueryBatcher::HEADER_SIZE + EncoderQueryBatcher::METADATA_SIZE * _queries.size();

    for(const auto& query : _queries) {
        const query_id_t& query_id = std::get<0>(query);
        const std::string& query_text = *std::get<2>(query);

        uint32_t query_text_size = mutils::bytes_size(query_text);

        total_text_size_bytes += query_text_size;
        total_size_bytes += query_text_size;

        _text_size_mapping[query_id] = query_text_size;
    }

    // use blob generator constructor
    // blob format:
    // [HEADER] - number of queries
    // [STRUCT] 8 bytes for query id, 4 bytes for client id, 4 bytes for strlen, STR BYTES
    _blob_repr = std::make_shared<derecho::cascade::Blob>([&](uint8_t* buffer, const size_t size){
        const uint32_t num_queries = _queries.size();

        // write the header
        std::memcpy(buffer, &num_queries, EncoderQueryBatcher::HEADER_SIZE);

        // write each query to the buffer
        uint32_t ptr_offset = EncoderQueryBatcher::HEADER_SIZE;
        for(const auto& query : _queries) {
            const query_id_t& query_id = std::get<0>(query);
            const uint32_t& client_id = std::get<1>(query);
            const char* text_ptr = std::get<2>(query)->c_str();
            const uint32_t& text_len = _text_size_mapping[query_id];

            // write query_id, client id, strlen
            uint32_t metadata_array[2] = {client_id, text_len};

            std::memcpy(buffer + ptr_offset, &query_id, sizeof(query_id_t));
            std::memcpy(buffer + ptr_offset + sizeof(query_id_t), metadata_array, sizeof(metadata_array));
            ptr_offset += EncoderQueryBatcher::METADATA_SIZE;

            // write str
            std::memcpy(buffer + ptr_offset, text_ptr, text_len);
            ptr_offset += text_len;
        } 
        return size;
    }, total_size_bytes);
}
