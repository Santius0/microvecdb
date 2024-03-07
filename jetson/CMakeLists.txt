cmake_minimum_required(VERSION 3.27)
project(microvecdb-jetson-nano-2gb)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-std=c++17)

set(CMAKE_SYSROOT "${CMAKE_CURRENT_SOURCE_DIR}/jetson-nano-sysroot")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
#set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
#set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")
#available flags: -march=armv8-a -mfloat-abi=hard -mfpu=neon -mfpu=crypto-neon-fp-armv8
set(CMAKE_C_FLAGS "-march=armv8-a")
set(CMAKE_CXX_FLAGS "-march=armv8-a")

function(include_sub_directories_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
        message("Including directory: ${root_dir}")
        include_directories(${root_dir})
        file(GLOB sub_dirs LIST_DIRECTORIES true RELATIVE ${root_dir} ${root_dir}/*)
        foreach(sub_dir ${sub_dirs})
            if (IS_DIRECTORY ${root_dir}/${sub_dir})
                include_sub_directories_recursively(${root_dir}/${sub_dir})
            endif()
        endforeach()
    endif()
endfunction()

include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

#file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/*.cc")
#list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc
add_executable(my_jetson_app
        ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/quantization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/preprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/index/flat_index.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/../src/storage/storage.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/../src/vectorizer/vectorizer.cpp
#        ${FASTTEXT_SOURCES}
)
target_include_directories(my_jetson_app PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(my_jetson_app pthread
#        rocksdb
#        Eigen3::Eigen
)