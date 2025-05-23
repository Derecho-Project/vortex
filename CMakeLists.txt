cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(vortex CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(CURL REQUIRED)
find_package(spdlog 1.3.1 REQUIRED)
find_package(derecho CONFIG REQUIRED)
find_package(cascade CONFIG REQUIRED)

# Find Faiss package
#find_package(Faiss REQUIRED)
# Find CUDA Toolkit if using Faiss with GPU support
#find_package(CUDAToolkit REQUIRED)
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
    ${HNSWLIB_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)



add_executable(start_webservice webservice/start_webservice.cpp webservice/webservice.cpp webservice/vortex_webservice.cpp webservice/cascade_client.cpp vortex_udls/serialize_utils.cpp)
target_link_libraries(start_webservice ${Boost_LIBRARIES} derecho::cascade)

add_subdirectory(benchmark)
add_subdirectory(vortex_udls)
add_subdirectory(data)

# Copy the configurations after the executables are built
add_dependencies(start_webservice benchmark_post_build python_udl_manager build_hnsw_index)
add_custom_command(TARGET start_webservice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
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
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/python_udls
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/vortex_udls/python/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/python_udls
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
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/data/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/data/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/data/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/data/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/data/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/perf_data
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/setup
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/setup
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/setup
    # COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/pipeline_client
    #     ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/pipeline_client
    # COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/pipeline_client
    #     ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/pipeline_client
    # COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/pipeline_client
    #     ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/pipeline_client
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/run_benchmark
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/run_benchmark
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/run_benchmark
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/run_benchmark
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/benchmark/run_benchmark
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/run_benchmark
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/start_webservice
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/start_webservice
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/start_webservice
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/start_webservice
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/start_webservice
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/start_webservice
    COMMAND chmod 755 ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/client_query.py
    COMMENT "prepare vortex udls and group configuration"
)
