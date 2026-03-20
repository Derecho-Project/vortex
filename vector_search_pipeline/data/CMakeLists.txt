add_executable(build_hnsw_index hnsw/build_hnsw_index.cpp)
target_include_directories(build_hnsw_index PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(build_hnsw_index Boost::python Boost::numpy ${Python_LIBRARIES})

add_custom_target(data_post_build ALL
    # link the data folder to the build directory
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/perf_data
        ${CMAKE_CURRENT_BINARY_DIR}/perf_data

    # link the python scripts to the build directory
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/balanced_ann.py
        ${CMAKE_CURRENT_BINARY_DIR}/balanced_ann.py
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/build_balanced_gpann.py
        ${CMAKE_CURRENT_BINARY_DIR}/build_balanced_gpann.py
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/faiss_knn.py
        ${CMAKE_CURRENT_BINARY_DIR}/faiss_knn.py
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/format_gist.py
        ${CMAKE_CURRENT_BINARY_DIR}/format_gist.py
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/gist_process_utils.py
        ${CMAKE_CURRENT_BINARY_DIR}/gist_process_utils.py
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/gp-ann
        ${CMAKE_CURRENT_BINARY_DIR}/gp-ann
)

add_dependencies(data_post_build build_hnsw_index)