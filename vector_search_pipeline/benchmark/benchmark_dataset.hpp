#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <filesystem>
#include <iostream>
#include <fstream>

#define QUERY_FILENAME "query.csv"
#define QUERY_EMB_FILENAME "query_emb.csv"
#define GROUNDTRUTH_FILENAME "groundtruth.csv"

class VortexBenchmarkDataset {
    void read_queries(uint64_t num_queries);
    void read_query_embs();
    void read_groundtruth();

    uint64_t next_query = 0;
    uint64_t emb_dim = 1024;
    bool groundtruth_loaded = false;
    std::string dataset_dir;
    std::vector<std::shared_ptr<std::string>> queries;
    std::vector<std::shared_ptr<float>> query_embs;
    std::vector<std::vector<std::string>> query_groundtruth;

    public:
        VortexBenchmarkDataset(const std::string& dataset_dir,uint64_t num_queries,uint64_t emb_dim);
        
        uint64_t get_next_query_index();
        void reset(){ next_query = 0; }
        bool has_groundtruth(){ return groundtruth_loaded; }

        const std::vector<std::string>& get_groundtruth(uint64_t query_index);
        std::shared_ptr<std::string> get_query(uint64_t query_index);
        std::shared_ptr<float> get_embeddings(uint64_t query_index);
};

