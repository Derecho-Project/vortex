cmake_minimum_required(VERSION 3.12.0)


if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CTest)

find_package(Catch2 2 REQUIRED)
find_package(benchmark REQUIRED)

add_executable(arena_tests
  tests/arena_tests.cpp
)

target_compile_features(arena_tests PRIVATE cxx_std_20)
target_include_directories(arena_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(arena_tests PRIVATE Catch2::Catch2WithMain)

add_executable(message_codec_tests
  tests/message_codec_tests.cpp
)

target_compile_features(message_codec_tests PRIVATE cxx_std_20)
target_include_directories(message_codec_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(message_codec_tests PRIVATE Catch2::Catch2WithMain vortex_scheduler)

add_executable(dag_registry_tests
  tests/dag_registry_tests.cpp
)

target_compile_features(dag_registry_tests PRIVATE cxx_std_20)
target_include_directories(dag_registry_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dag_registry_tests PRIVATE Catch2::Catch2WithMain vortex_scheduler)

add_executable(join_table_tests
  tests/join_table_tests.cpp
)

add_executable(task_join_service_tests
  tests/task_join_service_tests.cpp
)

add_executable(decision_gate_tests
  tests/decision_gate_tests.cpp
)

target_compile_features(join_table_tests PRIVATE cxx_std_20)
target_include_directories(join_table_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(join_table_tests PRIVATE Catch2::Catch2WithMain vortex_scheduler)

target_compile_features(task_join_service_tests PRIVATE cxx_std_20)
target_include_directories(task_join_service_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/example_scheduler
)
target_link_libraries(task_join_service_tests PRIVATE Catch2::Catch2WithMain vortex_scheduler)

target_compile_features(decision_gate_tests PRIVATE cxx_std_20)
target_include_directories(decision_gate_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(decision_gate_tests PRIVATE Catch2::Catch2WithMain vortex_scheduler)

include(Catch)
catch_discover_tests(arena_tests)
catch_discover_tests(message_codec_tests)
catch_discover_tests(dag_registry_tests)
catch_discover_tests(join_table_tests)
catch_discover_tests(task_join_service_tests)
catch_discover_tests(decision_gate_tests)

add_executable(arena_benchmarks
  benchmarks/arena_bench.cpp
)

target_compile_features(arena_benchmarks PRIVATE cxx_std_20)
target_include_directories(arena_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(arena_benchmarks PRIVATE benchmark::benchmark)

add_library(vortex_scheduler
  src/messages.cpp
  src/join_table.cpp
  src/dag_registry.cpp
  src/task_join_service.cpp
)
target_compile_features(vortex_scheduler PUBLIC cxx_std_20)
target_include_directories(vortex_scheduler PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(vortex_scheduler PUBLIC derecho::cascade)
