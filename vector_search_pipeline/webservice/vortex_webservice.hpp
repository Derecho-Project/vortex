
#include "cascade_client.hpp"
#include "webservice.hpp"

#define VORTEX_MAX_WAIT_TIME 60

class VortexWebService {
    uint64_t emb_dim;
    uint64_t batch_min_size;
    uint64_t batch_max_size;
    uint64_t batch_time_us;
    uint64_t num_result_threads;
    int num_io_threads;
    std::string address;
    unsigned short port;

public:
    VortexWebService(
            uint64_t emb_dim,
            uint64_t batch_min_size,
            uint64_t batch_max_size,
            uint64_t batch_time_us,
            uint64_t num_result_threads,
            int num_io_threads,
            std::string address,
            unsigned short port
        ) : emb_dim(emb_dim),
            batch_min_size(batch_min_size),
            batch_max_size(batch_max_size),
            batch_time_us(batch_time_us),
            num_result_threads(num_result_threads),
            num_io_threads(num_io_threads),
            address(address),
            port(port) {}

    void run();
};

