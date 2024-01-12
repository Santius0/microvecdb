cmake_minimum_required(VERSION 3.26)
project(microvecdb-tests)
set(CMAKE_CXX_STANDARD 17)

add_executable(tests tests.cpp)
target_link_libraries(tests gtest_main microvecdb)
add_test(NAME index_tests COMMAND bash -c "./index_tests --gtest_output='xml:${CMAKE_SOURCE_DIR}/tests/results/index_tests_results.xml'")