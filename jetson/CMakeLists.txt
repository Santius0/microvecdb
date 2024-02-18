cmake_minimum_required(VERSION 3.26)
project(microvecdb-jetson-nano-2gb)

#set(PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/..")
#set(CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/..")
set(CMAKE_SYSROOT "${CMAKE_SOURCE_DIR}/tegra_fs/filesystem")
#message(FATAL_ERROR ${CMAKE_SYSROOT})

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
#set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
#set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")
set(CMAKE_C_FLAGS "-march=armv8-a")
#set(CMAKE_C_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=neon")
#set(CMAKE_C_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8")
#set(CMAKE_CXX_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=neon")
set(CMAKE_CXX_FLAGS "-march=armv8-a")
#set(CMAKE_CXX_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8")
set(CMAKE_CXX_STANDARD 17)

#include(FetchContent)
#
#FetchContent_Declare(
#        spdlog
#        GIT_REPOSITORY https://github.com/gabime/spdlog.git
#        GIT_TAG v1.13.0
#)
#FetchContent_MakeAvailable(spdlog)

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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

add_executable(my_jetson_app
        ${CMAKE_CURRENT_SOURCE_DIR}/../main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/quantization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/preprocess.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/server.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/wal.cpp
)
target_include_directories(my_jetson_app PRIVATE ${eigen_SOURCE_DIR})
target_link_directories(my_jetson_app PRIVATE Eigen3::Eigen)

#target_compile_options(my_jetson_app PRIVATE -march=armv8-a)