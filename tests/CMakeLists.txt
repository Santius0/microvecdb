cmake_minimum_required(VERSION 3.26)
project(microvecdb-tests)
set(CMAKE_CXX_STANDARD 17)

add_executable(tests tests.cpp)
target_link_libraries(tests gtest_main microvecdb)
add_test(NAME db_tests COMMAND bash -c "./tests --gtest_output='xml:${CMAKE_SOURCE_DIR}/tests/results/tests_results.xml'")

add_executable(index_tests index/flat_index_tests.cpp index/faiss_flat_index_tests.cpp)
target_link_libraries(index_tests gtest_main microvecdb)
add_test(NAME index_tests COMMAND bash -c "./index_tests --gtest_output='xml:${CMAKE_SOURCE_DIR}/tests/results/index_tests_results.xml'")
