# Client log tags
set(LOG_TAG_QUERIES_SENDING_START 10000)
set(LOG_TAG_QUERIES_SENDING_END 10001)
set(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED 10100)

add_executable(run_benchmark benchmark_client.cpp 
                              run_benchmark.cpp 
                              benchmark_dataset.cpp 
                              ${CMAKE_SOURCE_DIR}/vortex_udls/serialize_utils.cpp)
target_link_libraries(run_benchmark derecho::cascade)
target_compile_definitions(run_benchmark PRIVATE
    LOG_TAG_QUERIES_SENDING_START=${LOG_TAG_QUERIES_SENDING_START}
    LOG_TAG_QUERIES_SENDING_END=${LOG_TAG_QUERIES_SENDING_END}
    LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED=${LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED}
)

add_custom_target(benchmark_post_build ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/setup
        ${CMAKE_CURRENT_BINARY_DIR}/setup
    # COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_client
    #     ${CMAKE_CURRENT_BINARY_DIR}/pipeline_client
)
add_dependencies(benchmark_post_build run_benchmark)