#include "vortex_client.hpp"

VortexPerfClient::VortexPerfClient(int node_id, int num_queries, int batch_size, int query_interval, int emb_dim): 
               my_node_id(node_id), num_queries(num_queries), batch_size(batch_size), query_interval(query_interval), embedding_dim(emb_dim) {
     this->running.store(true);
     this->num_queries_to_send.store(this->num_queries*this->batch_size);
}


int VortexPerfClient::read_queries(std::filesystem::path query_filepath, std::vector<std::string>& queries) {
     std::ifstream file(query_filepath);
     if (!file.is_open()) {
          std::cerr << "Error: Could not open query directory:" << query_filepath << std::endl;
          std::cerr << "Current only support query_doc in csv format." << std::endl;
          return 0;
     }
     std::string line;
     int num_query_collected = 0;
     while (std::getline(file, line)) {
          if (num_query_collected >= this->num_queries * this->batch_size) {
               break;
          }
          queries.push_back(line);
          num_query_collected++;
     }
     file.close();
     return num_query_collected;
}

int VortexPerfClient::read_query_embs(std::string query_emb_directory, std::unique_ptr<float[]>& query_embs){
     std::unique_ptr<float[]> embs = std::make_unique<float[]>(this->embedding_dim * this->num_queries * this->batch_size);
     std::ifstream file(query_emb_directory);
     if (!file.is_open()) {
          std::cerr << "Error: Could not open query directory:" << query_emb_directory << std::endl;
          std::cerr << "Current only support query_doc in csv format." << std::endl;
          return 0;
     }
     std::string line;
     int num_query_collected = 0;
     while (std::getline(file, line)) {
          if (num_query_collected >= this->num_queries * this->batch_size) {
               break;
          }
          std::istringstream ss(line);
          std::string token;
          int i = 0;
          while (std::getline(ss, token, ',')) {
               embs[num_query_collected * this->embedding_dim + i] = std::stof(token);
               i++;
          }
          if (i != this->embedding_dim) {
               std::cerr << "Error: query embedding dimension does not match." << std::endl;
               return 0;
          }
          num_query_collected++;
     }
     file.close();
     // resize the embs array to the actual number of queries collected
     if (num_query_collected < this->num_queries * this->batch_size) {
          std::unique_ptr<float[]> new_embs = std::make_unique<float[]>(this->embedding_dim * num_query_collected);
          std::memcpy(new_embs.get(), embs.get(), this->embedding_dim * num_query_collected * sizeof(float));
          embs = std::move(new_embs);
     }
     query_embs = std::move(embs);
     return num_query_collected;
}

std::string VortexPerfClient::format_query_emb_object(int nq, std::unique_ptr<float[]>& xq, std::vector<std::string>& query_list) {
     // create an bytes object by concatenating: num_queries + float array of emebddings + list of query_text
     uint32_t num_queries = static_cast<uint32_t>(nq);
     std::string nq_bytes(4, '\0');
     nq_bytes[0] = (num_queries >> 24) & 0xFF;
     nq_bytes[1] = (num_queries >> 16) & 0xFF;
     nq_bytes[2] = (num_queries >> 8) & 0xFF;
     nq_bytes[3] = num_queries & 0xFF;
     float* query_embeddings = xq.get();
     // serialize the query embeddings and query texts, formated as num_queries + query_embeddings + query_texts
     std::string query_emb_string = nq_bytes +
                              std::string(reinterpret_cast<const char*>(query_embeddings), sizeof(float) * this->embedding_dim * num_queries) +
                              nlohmann::json(query_list).dump();
     return query_emb_string;
}

bool VortexPerfClient::deserialize_result(const Blob& blob, std::string& query_text, std::vector<std::string>& top_k_docs,uint32_t& query_batch_id) {
     if (blob.size == 0) {
          std::cerr << "Error: empty result blob." << std::endl;
          return false;
     }
     char* json_data = const_cast<char*>(reinterpret_cast<const char*>(blob.bytes));
     std::size_t json_size = blob.size;
     std::string json_string(json_data, json_size);
     try{
          nlohmann::json parsed_json = nlohmann::json::parse(json_string);
          if (parsed_json.count("query") == 0 || parsed_json.count("top_k_docs") == 0) {
               std::cerr << "Result JSON does not contain query or top_k_docs." << std::endl;
               return false;
          }
          query_text = parsed_json["query"];
          top_k_docs = parsed_json["top_k_docs"];
          query_batch_id = parsed_json["query_batch_id"];

     } catch (const nlohmann::json::parse_error& e) {
          std::cerr << "Result JSON parse error: " << e.what() << std::endl;
          return false;
     }
     return true;
}

int VortexPerfClient::register_notification_on_all_servers(ServiceClientAPI& capi){
     // 1.1. Prepare for the notification by creating object pool for results to store
     std::string result_pool_name = "/rag/results/" + std::to_string(this->my_node_id);
     auto res = capi.template create_object_pool<VolatileCascadeStoreWithStringKey>(result_pool_name,0,HASH,{});
     for (auto& reply_future:res.get()) {
          reply_future.second.get(); // wait for the object pool to be created
     }
     std::cout << "Created object pool for results: " << result_pool_name << std::endl;
     // 1.2. Register notification for this object pool
     bool ret = capi.register_notification_handler(
               [&](const Blob& result){
                    std::string query_text;
                    std::vector<std::string> top_k_docs;
                    uint32_t query_batch_id;
                    if (!deserialize_result(result, query_text, top_k_docs, query_batch_id)) {
                         std::cerr << "Error: failed to deserialize the result from the notification." << std::endl;
                         return false;
                    }
                    if (this->query_results.find(query_text) == this->query_results.end()) {
                         this->query_results[query_text] = top_k_docs;
                    }
                    if (this->sent_queries.find(query_text) != this->sent_queries.end()) {
                         auto& tuple_vector = this->sent_queries[query_text];
                         if (!tuple_vector.empty()) {
                              auto first_tuple = tuple_vector.front();       
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
                         TimestampLogger::log(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED,this->my_node_id,std::get<0>(first_tuple),std::get<1>(first_tuple));
#endif
                              // std::cout << "Received result for query: " << query_text << " from client: " << this->my_node_id << " batch_id: " << batch_id << " q_id: " << q_id << std::endl;
                              // remove batch_id from this->sent_queries
                              tuple_vector.erase(tuple_vector.begin());
                              if (tuple_vector.empty()) {
                                   this->sent_queries.erase(query_text);
                              }
                         }
                    } else {
                         std::cerr << "Error: received result for query that is not sent." << std::endl;
                    }
                    if (this->sent_queries.size() == 0 && num_queries_to_send.load() == 0) {
                         this->running = false;
                         std::cout << "Received all results. Set running to false" << std::endl;
                    }
                    return true;
               }, result_pool_name);
     if (ret) {
          std::cerr << "Replaced previous notification handler for obj_pool: " << result_pool_name << std::endl;
     }

     // 1.3. Establishing connections to all server nodes
     auto subgroup_members = capi.template get_subgroup_members<VolatileCascadeStoreWithStringKey>(AGG_SUBGROUP_INDEX);
     int num_shards = subgroup_members.size();
     std::string establish_key = "establish_connection";
     for (int i = 0; i < num_shards; i++) {
          ObjectWithStringKey obj;
          std::string control_value = "establish";
          obj.key = establish_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, AGG_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object has been put
          }
     }
     std::cout << "Registered server-side notifications."<< std::endl;
     return num_shards;
}

bool VortexPerfClient::run_perf_test(ServiceClientAPI& capi, std::string& query_directory){
     // 1. Prepare the query and query embeddings
     /** TODO: quite some copies in this process, not on critical path, but could be optimized. */
     std::filesystem::path query_pathname = std::filesystem::path(query_directory) / QUERY_FILENAME;
     std::vector<std::string> queries;
     int num_query_collected = read_queries(query_pathname, queries);
     if (queries.size() == 0) {
          std::cerr << "Error: failed to read queries from " << query_directory << std::endl;
          return false;
     }

     std::filesystem::path query_emb_pathname = std::filesystem::path(query_directory) / QUERY_EMB_FILENAME;
     std::unique_ptr<float[]> query_embs;
     int num_emb_collected = read_query_embs(query_emb_pathname, query_embs);
     // resize query or query_embs to make sure they correspond, i.e. have the same number of queries
     if (num_query_collected > num_emb_collected){ // truncate query_vector to match the number of query embeddings
          queries.resize(num_query_collected);
          num_query_collected = num_emb_collected;
     } else if (num_emb_collected > num_query_collected) { // truncate query_embs to match the number of queries
          std::unique_ptr<float[]> new_embs = std::make_unique<float[]>(this->embedding_dim * num_query_collected);
          std::memcpy(new_embs.get(), query_embs.get(), this->embedding_dim * num_emb_collected * sizeof(float));
          query_embs = std::move(new_embs);
          num_emb_collected = num_query_collected;
     }
     if (num_query_collected < this->batch_size){
          std::cerr << "Error: total number of queries in the dataset are not large enough for the batch size." << std::endl;
          return false;
     }

     // 2. send the queries to the cascade
     for (int batch_id = 0; batch_id < this->num_queries; batch_id++) {
          std::string key = "/rag/emb/centroids_search/client" + std::to_string(this->my_node_id) + "/qb" + std::to_string(batch_id);
          // 2.1. Prepare the query texts
          std::vector<std::string> cur_query_list;
          std::vector<int> cur_query_ids;
          int qb_start_loc = batch_id * this->batch_size;
          for (int j = 0; j < this->batch_size; ++j) {
               int pos = (qb_start_loc + j) % num_query_collected;
              cur_query_list.push_back(queries[pos]);
              cur_query_ids.push_back(pos);
              if (this->sent_queries.find(queries[pos]) == this->sent_queries.end()) {
                   this->sent_queries[queries[pos]] = std::vector<std::tuple<uint32_t,uint32_t>>();
              }
              this->sent_queries[queries[pos]].push_back(std::make_tuple((uint32_t)batch_id, (uint32_t)j));
          }
          // 2.2. Prepare query embeddings
          std::unique_ptr<float[]> query_embeddings(new float[this->embedding_dim * this->batch_size]);
          for (int j = 0; j < this->batch_size; ++j) {
               int pos = (qb_start_loc + j) % num_emb_collected;
               for (int k = 0; k < this->embedding_dim; ++k) {
                    query_embeddings[j * this->embedding_dim + k] = query_embs[pos * this->embedding_dim + k];
               }
          }
          // 2.3. format the query 
          std::string emb_query_string = format_query_emb_object(this->batch_size, query_embeddings, cur_query_list);
          ObjectWithStringKey emb_query_obj;
          emb_query_obj.key = key;
          emb_query_obj.blob = Blob(reinterpret_cast<const uint8_t*>(emb_query_string.c_str()), emb_query_string.size());
          // 2.4. send the object to the cascade
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < this->batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_START,this->my_node_id,batch_id,j);
          }
#endif
          capi.put_and_forget(emb_query_obj, false); // not only trigger the UDL, but also update state. TODO: Need more thinking here. 
#ifdef ENABLE_VORTEX_EVALUATION_LOGGING
          for (int j = 0; j < this->batch_size; ++j) {
               TimestampLogger::log(LOG_TAG_QUERIES_SENDING_END,this->my_node_id,batch_id,j);
          }
#endif
          num_queries_to_send -= this->batch_size;
          std::this_thread::sleep_for(std::chrono::microseconds(this->query_interval)); // TODO
          if (batch_id % 200 == 0) {
               std::cout << "Sent " << batch_id << " queries." << std::endl;
          }
     }
     
     std::cout << "Put all queries to cascade." << std::endl;
     while (this->running.load()) {
          std::this_thread::sleep_for(std::chrono::microseconds(this->query_interval/100));
     }
     std::cout << "Received all results." << std::endl;
     return true;
}

bool VortexPerfClient::flush_logs(ServiceClientAPI& capi, int num_shards){
     std::string flush_log_key = "/rag/emb/centroids_search/flush_logs";
     for (int i = 0; i < num_shards; i++){
          ObjectWithStringKey obj;
          std::string control_value = "flush";
          obj.key = flush_log_key;
          obj.blob = Blob(reinterpret_cast<const uint8_t*>(control_value.c_str()), control_value.size());
          // TODO: 
          auto res = capi.template put<VolatileCascadeStoreWithStringKey>(obj, VORTEX_SUBGROUP_INDEX, i, true);
          for (auto& reply_future:res.get()) {
               reply_future.second.get(); // wait for the object pool to be created
          }
     }
     std::cout << "Flushed logs to shards." << std::endl;
     TimestampLogger::flush("client_timestamp.dat");
     return true;
}

std::vector<std::vector<std::string>> VortexPerfClient::read_groundtruth(std::filesystem::path filename) {
    std::vector<std::vector<std::string>> groundtruth_data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return groundtruth_data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(cell);
        }
        groundtruth_data.push_back(row);
    }
    file.close();
    return groundtruth_data;
}


bool VortexPerfClient::compute_recall(ServiceClientAPI& capi, std::string& query_directory){
     std::filesystem::path groundtruth_pathname = std::filesystem::path(query_directory) / GROUNDTRUTH_FILENAME;
     std::vector<std::vector<std::string>> groundtruth = read_groundtruth(groundtruth_pathname);
     double total_recall = 0.0;
     // double recalls[this->num_queries];
     for (const auto& [query, results] : this->query_results) {
          uint query_index = stoi(query.substr(query.find_last_of(' ') + 1));
          if (query_index >= groundtruth.size()) {
               std::cerr << "Error: query index out of range." << std::endl;
               return false;
          }
          uint topk = results.size();
          uint found = 0;
          const auto& query_grondtruth = groundtruth[query_index];
          for (const auto& result : results) {
               std::string doc_index = result.substr(result.find_last_of('/') + 1); // index are written as string
               if (is_in_topk(query_grondtruth, doc_index, topk)) {
                    found++;
               }
          }
          total_recall += static_cast<double>(found) / topk;
          // recalls[query_index] = static_cast<double>(found) / topk;
     }
     double avg_recall = total_recall / (this->num_queries*this->batch_size);
     std::cout << "Avg Recall: " << avg_recall << std::endl;
     std::cout << "------------------------" << std::endl;
     // Could write out recalls to a file
     return true;
}