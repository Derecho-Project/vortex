
#include <iostream>
#include "vortex_webservice.hpp"

#define DEFAULT_BATCH_MIN_SIZE 0
#define DEFAULT_BATCH_MAX_SIZE 5
#define DEFAULT_BATCH_TIME_US 500
#define DEFAULT_DIMENSIONS 1024
#define DEFAULT_NUM_RESULT_THREADS 1
#define DEFAULT_IO_THREADS 1
#define DEFAULT_ADDRESS "0.0.0.0"
#define DEFAULT_PORT 8080

void print_help(const std::string& bin_name){
    std::cout << "usage: " << bin_name << " [options]" << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << " -e <emb_dim>\t\t\tembeddings dimensions (default: " << DEFAULT_DIMENSIONS << ")" << std::endl; 
    std::cout << " -b <batch_min_size>\t\tminimum batch size (default: " << DEFAULT_BATCH_MIN_SIZE << ")" << std::endl;
    std::cout << " -x <batch_max_size>\t\tmaximum batch size (default: " << DEFAULT_BATCH_MAX_SIZE << ")" << std::endl;
    std::cout << " -u <batch_time_us>\t\tmaximum time to wait for the batch minimum size, in microseconds (default: " << DEFAULT_BATCH_TIME_US << ")" << std::endl;
    std::cout << " -t <num_result_threads>\tnumber of threads for processing results (default: " << DEFAULT_NUM_RESULT_THREADS << ")" << std::endl;
    std::cout << " -i <num_io_threads>\t\tnumber of threads for handling http requests (default: " << DEFAULT_IO_THREADS << ")" << std::endl;
    std::cout << " -a <address>\t\t\tIP address to bind to (default: " << DEFAULT_ADDRESS << ")" << std::endl; 
    std::cout << " -p <port>\t\t\tport to listen at (default: " << DEFAULT_PORT << ")" << std::endl; 
    std::cout << " -h\t\t\t\tshow this help" << std::endl;
}

int main(int argc, char** argv){
    char c;
    uint64_t batch_min_size = DEFAULT_BATCH_MIN_SIZE;
    uint64_t batch_max_size = DEFAULT_BATCH_MAX_SIZE;
    uint64_t batch_time_us = DEFAULT_BATCH_TIME_US;
    uint64_t emb_dim = DEFAULT_DIMENSIONS;
    uint64_t num_result_threads = DEFAULT_NUM_RESULT_THREADS;
    int num_io_threads = DEFAULT_IO_THREADS;
    std::string address = DEFAULT_ADDRESS;
    unsigned short port = DEFAULT_PORT;

    while ((c = getopt(argc, argv, "e:b:x:u:t:i:a:p:h")) != -1){
        switch(c){
            case 'e':
                emb_dim = strtoul(optarg,NULL,10);
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
            case 'i':
                num_io_threads = strtoul(optarg,NULL,10);
                break;
            case 'a':
                address = optarg;
                break;
            case 'p':
                port = strtoul(optarg,NULL,10);
                break;
            case '?':
            case 'h':
            default:
                print_help(argv[0]);
                return 0;
        }
    }

    std::cout << "starting Vortex web service ..." << std::endl;
    std::cout << "  emb_dim = " << emb_dim << std::endl;
    std::cout << "  batch_min_size = " << batch_min_size << std::endl;
    std::cout << "  batch_max_size = " << batch_max_size << std::endl;
    std::cout << "  batch_time_us = " << batch_time_us << std::endl;
    std::cout << "  num_result_threads = " << num_result_threads << std::endl;
    std::cout << "  num_io_threads = " << num_io_threads << std::endl;
    std::cout << "  address = " << address << std::endl;
    std::cout << "  port = " << port << std::endl;

    VortexWebService vortex(emb_dim,batch_min_size,batch_max_size,batch_time_us,num_result_threads,num_io_threads,address,port);
    vortex.run();
    return 0;
}

