cmake_minimum_required(VERSION 3.27)
project(microvecdb-jetson-nano-2gb)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT "${CMAKE_CURRENT_SOURCE_DIR}/jetson-nano-sysroot")
message(WARNING "${CMAKE_SYSROOT}")
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

#available flags: -march=armv8-a -mfloat-abi=hard -mfpu=neon -mfpu=crypto-neon-fp-armv8
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE) # neon is included by default in the armv8-a architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE)

#find_package(benchmark REQUIRED)

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

include_directories("${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/atlas")
link_directories("${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu" "${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu/atlas")
include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../include)
#include_sub_directories_recursively(${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../eigen)

#file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/*.cc")
#list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc

add_library(microvecdb ${CMAKE_CURRENT_SOURCE_DIR}/../src/index/distances.cpp)

#target_include_directories(microvecdb PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(microvecdb PUBLIC pthread openblas)

#add_executable(distance_metrics_benchmarks ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/index/distance_metrics_benchmarks.cpp)
#target_link_libraries(distance_metrics_benchmarks microvecdb benchmark::benchmark)