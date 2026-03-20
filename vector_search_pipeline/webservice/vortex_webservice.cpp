
#include "vortex_webservice.hpp"

void VortexWebService::run(){
    std::cout << "Starting Vortex Cascade client ..." << std::endl;
    VortexCascadeClient vortex;
    vortex.setup(batch_min_size,batch_max_size,batch_time_us,emb_dim,num_result_threads);

    // handler for HTTP requests
    request_handler_t request_handler = [&](http::request<http::string_body>&& req){
        uint64_t dim = req.body().size() / sizeof(float);

        // POST method
        if ((req.method() == http::verb::post) && (dim == emb_dim)) {
            // send query to vortex and wait for the result

            std::shared_ptr<float> emb(const_cast<float*>(reinterpret_cast<const float*>(req.body().c_str())),[](float* ptr){});
            auto fut = vortex.query_ann(emb);

            if(fut.wait_for(std::chrono::seconds(VORTEX_MAX_WAIT_TIME)) != std::future_status::timeout){
                // build the response
                http::response<http::string_body> res{http::status::ok, req.version()};
                res.set(http::field::server, "Vortex");
                res.set(http::field::content_type, "text/plain");
                res.keep_alive(req.keep_alive());

                std::shared_ptr<VortexANNResult> result = fut.get();
                const long * doc_ids = result->get_ids_pointer();
                uint32_t doc_ids_size = result->get_top_k();

                // a better way would be to send the raw array
                std::string body = std::to_string(doc_ids[0]);
                for(uint32_t i=1;i<doc_ids_size;i++){
                    body.append("@");
                    body.append(std::to_string(doc_ids[i]));
                }
                   
                res.body() = std::move(body);
                res.prepare_payload();
                return res;
            } else {
                std::cerr << "Timeout while waiting for future" << std::endl;
            }
        }

        // default response
        http::response<http::string_body> res{http::status::bad_request, req.version()};
        res.set(http::field::server, "Vortex");
        res.set(http::field::content_type, "text/plain");
        res.keep_alive(req.keep_alive());
        res.body() = "Method must be POST and embeddings dimension must be " + std::to_string(emb_dim) + " (received: " + std::to_string(dim) + ")";
        res.prepare_payload();
        return res;
    };
    
    std::cout << "Starting to listen to requests ..." << std::endl;
    auto const net_address = net::ip::make_address(address);
    try {
        net::io_context ioc{num_io_threads};
        auto listener = std::make_shared<Listener>(ioc, btcp::endpoint{net_address, port},request_handler);
        listener->run();

        // multi-threaded listening
        std::vector<std::thread> threads;
        threads.reserve(num_io_threads-1);

        for(auto i=num_io_threads-1; i>0; i--){
            threads.emplace_back([&ioc]{
                ioc.run();
            });
        }

        ioc.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

