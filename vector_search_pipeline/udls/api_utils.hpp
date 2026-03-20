#pragma once
#include <string>
#include <vector>

namespace api_utils {

bool get_batch_embeddings(const std::vector<std::string> &queries, 
                          const std::string &model, 
                          const std::string &api_key,
                          const int &dim,
                          float* all_embeddings);
std::string run_gpt4o_mini(const std::string &query, const std::vector<std::string> &top_docs, const std::string &model, const std::string &api_key);

}
