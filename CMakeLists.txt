cmake_minimum_required(VERSION 3.26)
project(confluxdb)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)

add_executable(confluxdb main.cpp
        src/RocksDBDatabase.cpp
        include/RocksDBDatabase.h)

target_link_libraries(confluxdb rocksdb)
