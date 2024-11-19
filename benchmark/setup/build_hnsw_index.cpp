#include "argparser.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <execution>
#include <thread>
#include <filesystem>
#include <fstream>
#include <hnswlib/hnswalg.h>
#include <iostream>
#include <memory>
#include <python3.10/Python.h>
#include <regex>
#include <sstream>
#include <vector>

constexpr size_t NUM_THREADS = 16;

namespace fs = std::filesystem;
namespace py = boost::python;
namespace np = boost::python::numpy;

std::vector<std::filesystem::path> find_all_files(const std::filesystem::path& dir,
                                                  std::function<bool(const std::string&, const std::string&)> pred) {
    std::list<std::filesystem::path> files_to_sweep;
    for(auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if(entry.is_regular_file()) {
            std::filesystem::path cur_file = entry.path();
            std::string type(cur_file.extension());
            if(pred(cur_file.stem().string(), type)) {
                files_to_sweep.push_back(std::move(cur_file));
            }
        }
    }
    return std::vector<std::filesystem::path>(std::make_move_iterator(files_to_sweep.begin()),
                                              std::make_move_iterator(files_to_sweep.end()));
}

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if(numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if(numThreads == 1) {
        for(size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        auto printer = [&end, &current]() {
            auto time_start = std::chrono::high_resolution_clock::now();
            while(true) {
                if(current >= end) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                // std::cout << "\r" << current << "                  ";
                auto curr = current.load();
                double progress = static_cast<double>(curr) / end;
                auto time_now = std::chrono::high_resolution_clock::now();
                auto time_elapsed = std::chrono::duration_cast<std::chrono::seconds>(time_now - time_start).count();

                auto it_per_s = static_cast<double>(curr) / static_cast<double>(time_elapsed);
                auto remaining_time = static_cast<int>(static_cast<double>(end - curr) / (static_cast<double>(it_per_s) + 0.001));

                fprintf(stdout, "[%7.2f] remaining time = %5ds, elapsed = %5lds  \r", (progress * 100), remaining_time, time_elapsed);
                fflush(stdout);
            }
        };
        std::thread progress(printer);

        for(size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while(true) {
                    size_t id = current.fetch_add(1);

                    if(id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch(...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }

        progress.join();
        for(auto& thread : threads) {
            thread.join();
        }
        if(lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

np::ndarray load_numpy_array(const fs::path& filename) {
    std::ifstream file(filename, std::ios::binary);
    if(!file) {
        throw std::runtime_error("Failed to open file: " + filename.string());
    }
    std::string pickle_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    py::object pickle = py::import("pickle");
    py::object data = pickle.attr("loads")(py::object(py::handle<>(PyBytes_FromStringAndSize(pickle_data.c_str(), pickle_data.size()))));
    return np::from_object(data);
}

template <typename T>
std::string vector_to_str(const std::vector<T> v) {
    if(v.size() == 0) {
        return std::string("[]");
    }

    std::stringstream ss;
    ss << '[' << v[0];
    for(size_t i = 1; i < v.size(); i++) {
        ss << ", " << v[i];
    }

    ss << ']';
    return ss.str();
}

template <typename T>
struct Embedding {
    const float* data;
    const size_t dim;
    const size_t nb;
};

void build_hnsw(hnswlib::HierarchicalNSW<float>& hnsw,
                const Embedding<float>& embedding) {
    const float* data = embedding.data;

    std::vector<std::vector<float>> normalized_point(NUM_THREADS, std::vector<float>(embedding.dim));

    ParallelFor(0, embedding.nb, NUM_THREADS, [&](size_t row, size_t id) {
        const float* point = data + embedding.dim * row;
        hnsw.addPoint(point, row);
    });
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program(argv[0]);
    program.add_description("directory should contain a centroids.pkl as well as cluster_n.pkl");
    program.add_argument("embedding_dir").help("path to directory benchmark/hnsw_index/<dataset_name>");
    program.add_argument("index_dir").help("path to directory to save the build indicies");
    program.add_argument("-m")
            .help("list of space separated m's used to build the HNSW")
            .scan<'i', int>()
            .nargs(argparse::nargs_pattern::at_least_one);
    program.add_argument("-e")
            .help("list of space separated ef's for construction")
            .scan<'i', int>()
            .nargs(argparse::nargs_pattern::at_least_one);

    try {
        program.parse_args(argc, argv);
    } catch(const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const fs::path embedding_dir{program.get<std::string>("embedding_dir")};
    const fs::path index_dir{program.get<std::string>("index_dir")};

    const std::vector<int> hyperparams_m = program.get<std::vector<int>>("-m");
    const std::vector<int> hyperparams_e = program.get<std::vector<int>>("-e");

    // check if the given directories exist
    if(!fs::exists(embedding_dir) || !fs::is_directory(embedding_dir)) {
        fprintf(stderr, "'%s' must exist and must be a directory\n", embedding_dir.string().c_str());
        return 1;
    }

    if(!fs::exists(index_dir) || !fs::is_directory(index_dir)) {
        fprintf(stderr, "'%s' must exist and must be a directory\n", index_dir.string().c_str());
        return 1;
    }

    // get clusters
    std::vector<fs::path> cluster_files = find_all_files(embedding_dir, [](const std::string& stem, const std::string& extension) {
        std::regex pattern("^cluster_\\d+$");
        return std::regex_match(stem, pattern) && extension == ".pkl";
    });

    fprintf(stdout, "Found cluster pickles:'\n");
    for(size_t i = 0; i < cluster_files.size(); i++) {
        fprintf(stdout, "\t%ld) %s\n", i + 1, cluster_files[i].string().c_str());
    }

    fprintf(stdout, "HNSW Building Settings\n");
    fprintf(stdout, "\t embedding path: '%s'\n", embedding_dir.string().c_str());
    fprintf(stdout, "\t index save path: '%s'\n", index_dir.string().c_str());

    fprintf(stdout, "\t M's to build: %s\n", vector_to_str(hyperparams_m).c_str());
    fprintf(stdout, "\t Ef construction's to build: %s\n", vector_to_str(hyperparams_e).c_str());

    // load numpy pickle file
    Py_Initialize();
    np::initialize();

    for(const auto& cluster_file : cluster_files) {
        np::ndarray array = load_numpy_array(cluster_file);
        const Py_intptr_t* shape = array.get_shape();
        const size_t num_vectors = static_cast<int>(shape[0]);
        const size_t dim = static_cast<int>(shape[1]);

        const std::string cluster_name = cluster_file.filename().stem().string();

        fprintf(stdout, "building cluster '%s' with dimensions: %ldx%ld\n", cluster_name.c_str(), num_vectors, dim);
        for(const int m : hyperparams_m) {
            for(const int ef_construction : hyperparams_e) {
                const fs::path save_file = index_dir / ("hnsw_m_" + std::to_string(m) + "_ef_" + std::to_string(ef_construction) + "_" + cluster_name + ".bin");

                if(fs::exists(save_file)) {
                    fprintf(stdout, "skipping index: '%s'\n", save_file.string().c_str());
                    continue;
                }
                fprintf(stdout, "generating index: '%s'\n", save_file.string().c_str());

                hnswlib::L2Space l2_space(dim);
                hnswlib::HierarchicalNSW<float> alg_hnsw = hnswlib::HierarchicalNSW<float>(&l2_space, num_vectors, m, ef_construction);
                const Embedding<float> embedding = {
                    static_cast<float*>(static_cast<void*>(array.get_data())),
                    dim,
                    num_vectors
                };

                build_hnsw(alg_hnsw, embedding);
                alg_hnsw.saveIndex(save_file.string());

                fprintf(stdout, "\n\n");
            }
        }
    }
    return 0;
}