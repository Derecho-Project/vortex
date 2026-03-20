#include "benchmark_dataset.hpp"

VortexBenchmarkDataset::VortexBenchmarkDataset(const std::string& dataset_dir,uint64_t num_queries,uint64_t emb_dim){
    this->dataset_dir = dataset_dir;
    this->emb_dim = emb_dim;

    read_queries(num_queries);
    read_query_embs();
    read_groundtruth();
}

uint64_t VortexBenchmarkDataset::get_next_query_index(){
    auto query_index = next_query;
    next_query++;
    return query_index;
}

std::shared_ptr<std::string> VortexBenchmarkDataset::get_query(uint64_t query_index){
    return queries[query_index % queries.size()];
}

std::shared_ptr<float> VortexBenchmarkDataset::get_embeddings(uint64_t query_index){
    return query_embs[query_index % query_embs.size()];
}

const std::vector<std::string>& VortexBenchmarkDataset::get_groundtruth(uint64_t query_index){
    return query_groundtruth[query_index % query_groundtruth.size()];
}

void VortexBenchmarkDataset::read_queries(uint64_t num_queries){
    std::filesystem::path query_filepath = std::filesystem::path(dataset_dir) / QUERY_FILENAME;

    std::ifstream file(query_filepath);
    if (!file.is_open()) {
        std::cerr << "  Error: Could not open query directory:" << query_filepath << std::endl;
        std::cerr << "  Current only support query_doc in csv format." << std::endl;
        return;
    }

    std::shared_ptr<std::string> line = std::make_shared<std::string>();
    while (std::getline(file, *line)) {
        if(queries.size() >= num_queries){
            break;
        }
        queries.push_back(line);
        line = std::make_shared<std::string>();
    }

    file.close();
}

void VortexBenchmarkDataset::read_query_embs(){
    std::filesystem::path query_emb_filepath = std::filesystem::path(dataset_dir) / QUERY_EMB_FILENAME;
    query_embs.reserve(queries.size());

    std::ifstream file(query_emb_filepath);
    if (!file.is_open()) {
        std::cerr << "  Error: Could not open query directory:" << query_emb_filepath << std::endl;
        std::cerr << "  Current only support query_doc in csv format." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if(query_embs.size() >= queries.size()){
            break;
        }

        float *embs = new float[emb_dim];
        std::istringstream ss(line);
        std::string token;
        int i = 0;
        while (std::getline(ss, token, ',')) {
            embs[i] = std::stof(token);
            i++;
        }

        if (i != emb_dim) {
            std::cerr << "  Error: query embedding dimension does not match." << std::endl;
            return;
        }

        std::shared_ptr<float> ptr(embs);
        query_embs.push_back(ptr);
    }

    file.close();

    if(query_embs.size() < queries.size()){
        std::cerr << "  Warning: number of embeddings (" << query_embs.size() << ") is smaller than number of queries (" << queries.size() << ")" << std::endl;
    }
}

void VortexBenchmarkDataset::read_groundtruth(){
    std::filesystem::path filename = std::filesystem::path(dataset_dir) / GROUNDTRUTH_FILENAME; 
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "  No groundtruth file found!" << std::endl;
        groundtruth_loaded = false;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(cell);
        }
        query_groundtruth.push_back(row);
    }

    file.close();
    groundtruth_loaded = true;
}

