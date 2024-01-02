cmake_minimum_required(VERSION 3.26)
project(microvecdb-tests)
set(CMAKE_CXX_STANDARD 17)

add_executable(sample index_tests.cpp)
target_link_libraries(sample gtest_main)
add_test(NAME sample_test COMMAND sample)