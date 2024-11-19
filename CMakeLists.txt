cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(vortex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(spdlog 1.3.1 REQUIRED)
find_package(derecho CONFIG REQUIRED)
find_package(cascade CONFIG REQUIRED)

# Find Faiss package
find_package(Faiss REQUIRED)
# Find CUDA Toolkit if using Faiss with GPU support
find_package(CUDAToolkit REQUIRED)
# Find Hnswlib package
find_package(hnswlib REQUIRED)

# Boost
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Boost REQUIRED COMPONENTS system filesystem numpy)
include_directories(${Boost_INCLUDE_DIRS})

include(GNUInstallDirs)
if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR bin)
endif()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()



include_directories(
    ${FAISS_INCLUDE_DIRS}
    ${HNSWLIB_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

link_directories("/root/opt-dev/lib")

# Client log tags
set(LOG_TAG_QUERIES_SENDING_START 10000)
set(LOG_TAG_QUERIES_SENDING_END 10001)
set(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED 10100)

add_executable(build_hnsw_index benchmark/setup/build_hnsw_index.cpp)
target_include_directories(build_hnsw_index PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(build_hnsw_index Boost::python Boost::numpy ${Python_LIBRARIES})

add_executable(run_benchmark benchmark/benchmark_client.cpp benchmark/run_benchmark.cpp benchmark/benchmark_dataset.cpp)
target_link_libraries(run_benchmark derecho::cascade)
target_compile_definitions(run_benchmark PRIVATE
    LOG_TAG_QUERIES_SENDING_START=${LOG_TAG_QUERIES_SENDING_START}
    LOG_TAG_QUERIES_SENDING_END=${LOG_TAG_QUERIES_SENDING_END}
    LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED=${LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED}
)

add_executable(latency_client benchmark/vortex_client.cpp benchmark/latency_client.cpp vortex_udls/serialize_utils.cpp)
target_link_libraries(latency_client PRIVATE libwsong::perf derecho derecho::cascade pthread OpenSSL::Crypto)
target_compile_definitions(latency_client PRIVATE
    LOG_TAG_QUERIES_SENDING_START=${LOG_TAG_QUERIES_SENDING_START}
    LOG_TAG_QUERIES_SENDING_END=${LOG_TAG_QUERIES_SENDING_END}
    LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED=${LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED}
)

add_executable(start_webservice webservice/start_webservice.cpp webservice/webservice.cpp webservice/vortex_webservice.cpp webservice/cascade_client.cpp)
target_link_libraries(start_webservice ${Boost_LIBRARIES} derecho::cascade)

set(UDL_COMMON_LIBS derecho derecho::cascade pthread faiss CUDA::cudart)

# Centroids_search UDL tags
set(LOG_BATCH_FAISS_SEARCH_START 30030)
set(LOG_BATCH_FAISS_SEARCH_END 30031)
set(LOG_BATCH_FAISS_SEARCH_SIZE 30032)

set(LOG_CENTROIDS_EMBEDDINGS_UDL_START 20000)
set(LOG_CENTROIDS_EMBEDDINGS_LOADING_START 20010)
set(LOG_CENTROIDS_EMBEDDINGS_LOADING_END 20011)
set(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_START 20020)
set(LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_END 20021)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START 20030)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END 20031)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END 20041)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START 20050)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END 20051)
set(LOG_CENTROIDS_EMBEDDINGS_UDL_END 20100)


# Cluster Search tags
set(LOG_CLUSTER_SEARCH_UDL_START 30000)
set(LOG_CLUSTER_SEARCH_UDL_LOADEMB_START 30010)
set(LOG_CLUSTER_SEARCH_UDL_LOADEMB_END 30011)
set(LOG_CLUSTER_SEARCH_DESERIALIZE_START 30020)
set(LOG_CLUSTER_SEARCH_DESERIALIZE_END 30021)
set(LOG_CLUSTER_SEARCH_ADDED_TOBATCH 30022)
set(LOG_CLUSTER_SEARCH_CONSTRUCT_KEYS_END 30041)
set(LOG_CLUSTER_SEARCH_UDL_EMIT_START 30050)


add_library(centroids_search_udl SHARED
                            vortex_udls/centroids_search_udl.hpp 
                            vortex_udls/centroids_search_udl.cpp 
                            vortex_udls/serialize_utils.cpp 
                            vortex_udls/api_utils.cpp)
target_link_libraries(centroids_search_udl PRIVATE ${UDL_COMMON_LIBS} CURL::libcurl)
target_compile_definitions(centroids_search_udl PRIVATE
    LOG_CENTROIDS_EMBEDDINGS_UDL_START=${LOG_CENTROIDS_EMBEDDINGS_UDL_START}
    LOG_CENTROIDS_EMBEDDINGS_LOADING_START=${LOG_CENTROIDS_EMBEDDINGS_LOADING_START}
    LOG_CENTROIDS_EMBEDDINGS_LOADING_END=${LOG_CENTROIDS_EMBEDDINGS_LOADING_END}
    LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_START=${LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_START}
    LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_END=${LOG_CENTROIDS_SEARCH_PREPARE_EMBEDDINGS_END}
    LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START=${LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_START}
    LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END=${LOG_CENTROIDS_EMBEDDINGS_UDL_SEARCH_END}
    LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END=${LOG_CENTROIDS_EMBEDDINGS_UDL_COMBINE_END}
    LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START=${LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_START}
    LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END=${LOG_CENTROIDS_EMBEDDINGS_UDL_EMIT_END}
    LOG_CENTROIDS_EMBEDDINGS_UDL_END=${LOG_CENTROIDS_EMBEDDINGS_UDL_END}
    LOG_BATCH_FAISS_SEARCH_SIZE=${LOG_BATCH_FAISS_SEARCH_SIZE}
    LOG_BATCH_FAISS_SEARCH_START=${LOG_BATCH_FAISS_SEARCH_START}
    LOG_BATCH_FAISS_SEARCH_END=${LOG_BATCH_FAISS_SEARCH_END}
)


add_library(clusters_search_udl SHARED 
                                vortex_udls/clusters_search_udl.hpp 
                                vortex_udls/clusters_search_udl.cpp 
                                vortex_udls/serialize_utils.cpp)
target_link_libraries(clusters_search_udl PRIVATE ${UDL_COMMON_LIBS})

target_compile_definitions(clusters_search_udl PRIVATE
    LOG_CLUSTER_SEARCH_UDL_START=${LOG_CLUSTER_SEARCH_UDL_START}
    LOG_CLUSTER_SEARCH_UDL_LOADEMB_START=${LOG_CLUSTER_SEARCH_UDL_LOADEMB_START}
    LOG_CLUSTER_SEARCH_UDL_LOADEMB_END=${LOG_CLUSTER_SEARCH_UDL_LOADEMB_END}
    LOG_CLUSTER_SEARCH_DESERIALIZE_START=${LOG_CLUSTER_SEARCH_DESERIALIZE_START}
    LOG_CLUSTER_SEARCH_DESERIALIZE_END=${LOG_CLUSTER_SEARCH_DESERIALIZE_END}
    LOG_CLUSTER_SEARCH_ADDED_TOBATCH=${LOG_CLUSTER_SEARCH_ADDED_TOBATCH}
    LOG_BATCH_FAISS_SEARCH_START=${LOG_BATCH_FAISS_SEARCH_START}
    LOG_BATCH_FAISS_SEARCH_END=${LOG_BATCH_FAISS_SEARCH_END}
    LOG_BATCH_FAISS_SEARCH_SIZE=${LOG_BATCH_FAISS_SEARCH_SIZE}
    LOG_CLUSTER_SEARCH_CONSTRUCT_KEYS_END=${LOG_CLUSTER_SEARCH_CONSTRUCT_KEYS_END}
    LOG_CLUSTER_SEARCH_UDL_EMIT_START=${LOG_CLUSTER_SEARCH_UDL_EMIT_START}
)

# AGG GENERATE UDL tags
set(LOG_TAG_AGG_UDL_START 40000)
set(LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE 40001)
set(LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED 40010)
set(LOG_TAG_AGG_UDL_QUERY_FINISHED_GATHERED 40011)
set(LOG_TAG_AGG_UDL_RETRIEVE_DOC_START 40020)
set(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START 40120)
set(LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END 40121)
set(LOG_TAG_AGG_UDL_LOAD_DOC_START 40220)
set(LOG_TAG_AGG_UDL_LOAD_DOC_END 40221)
set(LOG_TAG_AGG_UDL_RETRIEVE_DOC_END 40021)
set(LOG_TAG_AGG_UDL_PUT_RESULT_START 40030)
set(LOG_TAG_AGG_UDL_PUT_RESULT_END 40031)
set(LOG_TAG_AGG_UDL_END 40100)


add_library(aggregate_generate_udl SHARED 
            vortex_udls/aggregate_generate_udl.hpp 
            vortex_udls/aggregate_generate_udl.cpp 
            vortex_udls/serialize_utils.cpp 
            vortex_udls/api_utils.cpp)
target_link_libraries(aggregate_generate_udl PRIVATE derecho::cascade CURL::libcurl)

target_compile_definitions(aggregate_generate_udl PRIVATE
    LOG_TAG_AGG_UDL_END=${LOG_TAG_AGG_UDL_END}
    LOG_TAG_AGG_UDL_START=${LOG_TAG_AGG_UDL_START}
    LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE=${LOG_TAG_AGG_UDL_FINISHED_DESERIALIZE}
    LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED=${LOG_TAG_AGG_UDL_END_NOT_FULLY_GATHERED}
    LOG_TAG_AGG_UDL_QUERY_FINISHED_GATHERED=${LOG_TAG_AGG_UDL_QUERY_FINISHED_GATHERED}
    LOG_TAG_AGG_UDL_RETRIEVE_DOC_START=${LOG_TAG_AGG_UDL_RETRIEVE_DOC_START}
    LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START=${LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_START}
    LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END=${LOG_TAG_AGG_UDL_LOAD_EMB_DOC_MAP_END}
    LOG_TAG_AGG_UDL_LOAD_DOC_START=${LOG_TAG_AGG_UDL_LOAD_DOC_START}
    LOG_TAG_AGG_UDL_LOAD_DOC_END=${LOG_TAG_AGG_UDL_LOAD_DOC_END}
    LOG_TAG_AGG_UDL_RETRIEVE_DOC_END=${LOG_TAG_AGG_UDL_RETRIEVE_DOC_END}
    LOG_TAG_AGG_UDL_PUT_RESULT_START=${LOG_TAG_AGG_UDL_PUT_RESULT_START}
    LOG_TAG_AGG_UDL_PUT_RESULT_END=${LOG_TAG_AGG_UDL_PUT_RESULT_END}
    LOG_FAISS_SEARCH_BATCH_SIZE=${LOG_FAISS_SEARCH_BATCH_SIZE}
    LOG_TAG_AGG_UDL_END=${LOG_TAG_AGG_UDL_END}
)

add_custom_command(TARGET aggregate_generate_udl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/perf_data
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/perf_data
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/setup
        ${CMAKE_CURRENT_BINARY_DIR}/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/dfgs.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/clear_log.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/clear_log.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/clear_log.sh 
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/clear_log.sh
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/latency_client
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/latency_client
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/latency_client
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/latency_client
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/latency_client
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/latency_client
    COMMAND chmod 755 ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
    # symlink precomputed indices to the config folder of each node
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hnsw_index ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/hnsw_index
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hnsw_index ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/hnsw_index
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hnsw_index ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/hnsw_index
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/hnsw_index ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/hnsw_index
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/client_query.py
    COMMENT "prepare cluster search configuration"
)
