cmake_minimum_required(VERSION 3.27)
project(microvecdb-bencharks)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Enable testing of the benchmark library.")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Enable testing of the benchmark library." FORCE)
FetchContent_MakeAvailable(googlebenchmark)

#find_package(benchmark REQUIRED)


add_executable(distance_metrics_benchmarks index/distance_metrics_benchmarks.cpp)
target_link_libraries(distance_metrics_benchmarks microvecdb benchmark::benchmark pthread)