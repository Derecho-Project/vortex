#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <unistd.h>
#include "benchmark_client.hpp"
#include "benchmark_dataset.hpp"

#define DEFAULT_BATCH_MIN_SIZE 0
#define DEFAULT_BATCH_MAX_SIZE 5
#define DEFAULT_BATCH_TIME_US 500
#define DEFAULT_DIMENSIONS 1024
#define DEFAULT_NUM_QUERIES 10
#define DEFAULT_NUM_RESULT_THREADS 1
#define DEFAULT_WARMUP 0

void print_help(const std::string& bin_name){
    std::cout << "usage: " << bin_name << " [options] <dataset_dir>" << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << " -n <num_queries>\t\ttotal number of queries to send (default: " << DEFAULT_NUM_QUERIES << ")" << std::endl;
    std::cout << " -e <emb_dim>\t\t\tembeddings dimensions (default: " << DEFAULT_DIMENSIONS << ")" << std::endl;
    std::cout << " -r <send_rate>\t\t\trate (in queries/second) at which to send queries (default: unlimited)" << std::endl;
    std::cout << " -b <batch_min_size>\t\tminimum batch size (default: " << DEFAULT_BATCH_MIN_SIZE << ")" << std::endl;
    std::cout << " -x <batch_max_size>\t\tmaximum batch size (default: " << DEFAULT_BATCH_MAX_SIZE << ")" << std::endl;
    std::cout << " -u <batch_time_us>\t\tmaximum time to wait for the batch minimum size, in microseconds (default: " << DEFAULT_BATCH_TIME_US << ")" << std::endl;
    std::cout << " -t <num_result_threads>\tnumber of threads for processing results (default: " << DEFAULT_NUM_RESULT_THREADS << ")" << std::endl;
    std::cout << " -w <num_warmup_queries>\tsend the given number of queries to warm up the system, before starting the benchmark (default: " << DEFAULT_WARMUP << ")" << std::endl;
    std::cout << " -d\t\t\t\tdo not dump timestamps in the remote servers (default: true)" << std::endl;
    std::cout << " -h\t\t\t\tshow this help" << std::endl;
}

inline bool is_in_topk(const std::vector<std::string>& groundtruth, long doc_id, int k) {
    std::string target = std::to_string(doc_id);
    k = std::min(k, (int)groundtruth.size());
    auto it = std::find(groundtruth.begin(), groundtruth.begin() + k, target);
    return it != (groundtruth.begin() + k);
}

int main(int argc, char** argv){
    char c;
    uint64_t send_rate = 0;
    bool rate_control = false;
    uint64_t batch_min_size = DEFAULT_BATCH_MIN_SIZE;
    uint64_t batch_max_size = DEFAULT_BATCH_MAX_SIZE;
    uint64_t batch_time_us = DEFAULT_BATCH_TIME_US;
    uint64_t num_queries = DEFAULT_NUM_QUERIES;
    uint64_t emb_dim = DEFAULT_DIMENSIONS;
    uint64_t num_result_threads = DEFAULT_NUM_RESULT_THREADS;
    uint64_t num_warmup = DEFAULT_WARMUP;
    bool dump = true;

    while ((c = getopt(argc, argv, "n:e:r:b:x:u:t:w:dh")) != -1){
        switch(c){
            case 'n':
                num_queries = strtoul(optarg,NULL,10);
                break;
            case 'e':
                emb_dim = strtoul(optarg,NULL,10);
                break;
            case 'r':
                send_rate = strtoul(optarg,NULL,10);
                break;
            case 'b':
                batch_min_size = strtoul(optarg,NULL,10);
                break;
            case 'x':
                batch_max_size = strtoul(optarg,NULL,10);
                break;
            case 'u':
                batch_time_us = strtoul(optarg,NULL,10);
                break;
            case 't':
                num_result_threads = strtoul(optarg,NULL,10);
                break;
            case 'w':
                num_warmup = strtoul(optarg,NULL,10);
                break;
            case 'd':
                dump = false;
                break;
            case '?':
            case 'h':
            default:
                print_help(argv[0]);
                return 0;
        }
    }

    if(optind >= argc){
        print_help(argv[0]);
        return 0;
    }

    std::string dataset_dir(argv[optind]);
    std::chrono::nanoseconds iteration_time;
    if(send_rate != 0){
        rate_control = true;
        iteration_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::seconds(1)) / send_rate;
    }

    std::cout << "starting benchmark ..." << std::endl;
    std::cout << "  num_queries = " << num_queries << std::endl;
    std::cout << "  send_rate = " << send_rate << std::endl;
    std::cout << "  emb_dim = " << emb_dim << std::endl;
    std::cout << "  batch_min_size = " << batch_min_size << std::endl;
    std::cout << "  batch_max_size = " << batch_max_size << std::endl;
    std::cout << "  batch_time_us = " << batch_time_us << std::endl;
    std::cout << "  num_result_threads = " << num_result_threads << std::endl;
    std::cout << "  num_warmup_queries = " << num_warmup << std::endl;
    std::cout << "  dump = " << dump << std::endl;
    std::cout << "  dataset_dir = " << dataset_dir << std::endl;

    VortexBenchmarkDataset dataset(dataset_dir,num_queries,emb_dim);
    VortexBenchmarkClient vortex(num_warmup);

    // setup
    std::cout << "setting up client ..." << std::endl;
    vortex.setup(batch_min_size,batch_max_size,batch_time_us,emb_dim,num_result_threads);

    // warmup
    if(num_warmup > 0){
        std::cout << "warmup: sending " << num_warmup << " queries ..." << std::endl;
        auto warmup_start = std::chrono::steady_clock::now();
        auto extra_time = std::chrono::nanoseconds(0);
        for(uint64_t i=0;i<num_warmup;i++){
            auto start = std::chrono::steady_clock::now();
            uint64_t next_query_index = dataset.get_next_query_index();
            auto query_text = dataset.get_query(next_query_index);
            auto query_emb = dataset.get_embeddings(next_query_index);
            uint64_t query_id = vortex.query(query_text,query_emb);
            
            auto end = std::chrono::steady_clock::now();
            if(rate_control){
                auto elapsed = end - start + extra_time;
                auto sleep_time = iteration_time - elapsed;
                start = std::chrono::steady_clock::now();
                std::this_thread::sleep_for(sleep_time);
                extra_time = std::chrono::steady_clock::now() - start - sleep_time;
            }
        }
        
        std::cout << "  all sent, waiting results ..." << std::endl;
        vortex.wait_results();
        
        std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - warmup_start);
        std::cout << "  system warmed up in " << elapsed.count() << " ms" << std::endl;
        dataset.reset();
    }

    // send queries
    std::cout << "starting benchmark: sending " << num_queries << " queries ..." << std::endl;
    std::unordered_map<uint64_t,uint64_t> query_id_to_index;
    auto extra_time = std::chrono::nanoseconds(0);
    for(uint64_t i=0;i<num_queries;i++){
        auto start = std::chrono::steady_clock::now();
        if (i % 200 == 0){
            std::cout << "  sent " << i << std::endl;
        }

        uint64_t next_query_index = dataset.get_next_query_index();
        auto query_text = dataset.get_query(next_query_index);
        auto query_emb = dataset.get_embeddings(next_query_index);
        uint64_t query_id = vortex.query(query_text,query_emb);
        query_id_to_index[query_id] = next_query_index;
        
        auto end = std::chrono::steady_clock::now();
        if(rate_control){
            auto elapsed = end - start + extra_time;
            auto sleep_time = iteration_time - elapsed;
            start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(sleep_time);
            extra_time = std::chrono::steady_clock::now() - start - sleep_time;
        }
    }
    std::cout << "  all queries sent!" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // wait until all results are received
    std::cout << "waiting all results to arrive ..." << std::endl;
    vortex.wait_results();
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // recall
    if(dataset.has_groundtruth()){
        std::cout << "computing recall ..." << std::endl;
        double total_recall = 0.0;
        for (const auto& [query_id, query_index] : query_id_to_index){
            const auto& results = vortex.get_result(query_id);
            const auto& groundtruth = dataset.get_groundtruth(query_index);

            uint64_t topk = results->get_top_k();
            uint64_t found = 0;
            const long * doc_ids = results->get_ids_pointer();
            for (uint64_t i=0; i<topk; i++) {
                if (is_in_topk(groundtruth, doc_ids[i], topk)) {
                    found++;
                }
            }
            total_recall += static_cast<double>(found) / topk;
        }

        double avg_recall = total_recall / num_queries;
        std::cout << " avg recall: " << avg_recall << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    } else {
        std::cout << "no groundtruth available, skipping computing recall ..." << std::endl;
    }

    // write timestamps
    std::cout << "dumping timestamps ..." << std::endl;
    vortex.dump_timestamps(dump);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    return 0;
}

