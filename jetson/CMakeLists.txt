cmake_minimum_required(VERSION 3.27)
project(microvecdb-jetson-nano-2gb)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT "${CMAKE_CURRENT_SOURCE_DIR}/jetson-nano-sysroot")
#set(CMAKE_SYSROOT /home/santius/x-tools/aarch64-unknown-linux-gnu/aarch64-unknown-linux-gnu/sysroot)

message(STATUS "sysroot: ${CMAKE_SYSROOT}")

#set(CMAKE_C_COMPILER /home/santius/x-tools/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-gcc)
#set(CMAKE_CXX_COMPILER /home/santius/x-tools/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++)
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

#available flags: -march=armv8-a -mfloat-abi=hard -mfpu=neon -mfpu=crypto-neon-fp-armv8
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE) # neon is included by default in the armv8-a architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE)

set(CMAKE_STAGING_PREFIX "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# Adjust default behavior of the FIND_XXX commands to look for libraries and headers in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#find_package(benchmark REQUIRED)
#message(FATAL_ERROR $ENV{HOME})
#
#find_path(NUMA_INCLUDE_DIR NAME numa.h HINTS
#        ${CMAKE_SYSROOT}/local/include
#        ${CMAKE_SYSROOT}/opt/local/include
#        ${CMAKE_SYSROOT}/usr/local/include
#        ${CMAKE_SYSROOT}/usr/include # this one
#)
#
#find_library(NUMA_LIBRARY NAME libnuma.so HINTS
#        ${CMAKE_SYSROOT}/local/lib64
#        ${CMAKE_SYSROOT}/local/lib
#        ${CMAKE_SYSROOT}/usr/local/lib64
#        ${CMAKE_SYSROOT}/usr/local/lib
#        ${CMAKE_SYSROOT}/opt/local/lib64
#        ${CMAKE_SYSROOT}/opt/local/lib
#        ${CMAKE_SYSROOT}/usr/lib64
#        ${CMAKE_SYSROOT}/usr/lib
#        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
#)
#
#find_library(NUMA_LIBRARY_STATIC NAME libnuma.a HINTS
#        ${CMAKE_SYSROOT}/local/lib64
#        ${CMAKE_SYSROOT}/local/lib
#        ${CMAKE_SYSROOT}/usr/local/lib64
#        ${CMAKE_SYSROOT}/usr/local/lib
#        ${CMAKE_SYSROOT}/opt/local/lib64
#        ${CMAKE_SYSROOT}/opt/local/lib
#        ${CMAKE_SYSROOT}/usr/lib64
#        ${CMAKE_SYSROOT}/usr/lib
#        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu # this one
#)
#
#if (NUMA_INCLUDE_DIR AND NUMA_LIBRARY AND NUMA_LIBRARY_STATIC)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lnuma")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lnuma")
#    include_directories (${NUMA_INCLUDE_DIR})
#    message (STATUS "Found numa library: inc=${NUMA_INCLUDE_DIR}, lib=${NUMA_LIBRARY}, staticlib=${NUMA_LIBRARY_STATIC}")
#    add_definitions(-DNUMA)
#else ()
#    set (NUMA_LIBRARY "")
#    set (NUMA_LIBRARY_STATIC "")
#    message (STATUS "WARNING: Numa library not found.")
#    message (STATUS "Try: 'sudo yum install numactl numactl-devel' (or sudo apt-get install libnuma libnuma-dev)")
#endif ()


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

#find_package(OpenMP REQUIRED)

add_executable(jetson_test jetson_test.cpp
#        OpenMP::OpenMP_CXX OpenMP::OpenMP_C
)
target_link_directories(jetson_test PUBLIC ${PROJECT_SOURCE_DIR}/../numa_aarch64 ${PROJECT_SOURCE_DIR}/../zstd_aarch64)
target_include_directories(jetson_test PUBLIC ${PROJECT_SOURCE_DIR}/../numa_aarch64 ${PROJECT_SOURCE_DIR}/../zstd_aarch64)
target_link_libraries(jetson_test PUBLIC zstd pthread)

#include_directories("${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/atlas")
#link_directories("${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/atlas")
#include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

#file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/*.cc")
#list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc

#add_library(microvecdb ${CMAKE_CURRENT_SOURCE_DIR}/../src/index/distances.cpp)

#target_include_directories(microvecdb PRIVATE ${eigen_SOURCE_DIR})
#target_link_libraries(microvecdb PUBLIC pthread openblas OpenMP::OpenMP_CXX OpenMP::OpenMP_C)

#add_executable(distance_metrics_benchmarks ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/index/distance_metrics_benchmarks.cpp)
#target_link_libraries(distance_metrics_benchmarks microvecdb benchmark::benchmark)