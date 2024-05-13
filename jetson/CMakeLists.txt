cmake_minimum_required(VERSION 3.27)
project(microvecdb-jetson-nano-2gb)
set(CMAKE_CXX_STANDARD 17)

#if(CMAKE_BUILD_TYPE STREQUAL "Release")
#    message(STATUS "Build type: Release")
#    #    set(LIB_DIR ${PROJECT_SOURCE_DIR}/release/lib)
#elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    message(STATUS "Build type: Debug")
#    #    set(LIB_DIR ${PROJECT_SOURCE_DIR}/debug/lib)
#else()
#    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
#endif()

#set(BUILD_SHARED_LIBS ON)
#
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DANNOYLIB_MULTITHREADED_BUILD)

set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

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
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fPIC --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE) # neon is included by default in the armv8-a architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE)

set(CMAKE_STAGING_PREFIX "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# Adjust default behavior of the FIND_XXX commands to look for libraries and headers in the target environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Constructing Platform Tag >>>> ----------------------------------------------------------
set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS "linux-gnu")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(OS "win32")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(OS "darwin")
else()
    set(OS ${CMAKE_SYSTEM_NAME})
endif()
set(PLATFORM_TAG "${ARCHITECTURE}-${OS}")
message(STATUS "Platform Tag: ${PLATFORM_TAG}")
# <<<< Constructing Platform Tag  ----------------------------------------------------------

# Getting Python Major & Minor Version >>>> ----------------------------------------------------------
find_path(Python3_INCLUDE_DIRS NAME Python.h HINTS
        ${CMAKE_SYSROOT}/usr/include/python3.6m
        ${CMAKE_SYSROOT}/usr/include/python3.6
        ${CMAKE_SYSROOT}/usr/include
)

find_library(Python3_LIBRARIES NAME libpython3.6m.so HINTS
        ${CMAKE_SYSROOT}/usr/lib/python3.6/config-3.6m-aarch64-linux-gnu
        ${CMAKE_SYSROOT}/usr/lib/python3.6
        ${CMAKE_SYSROOT}/usr/local/lib/python3.6
        ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu
)

set(Python3_VERSION 3.6m)
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")
string(REPLACE "." ";" PYTHON_VERSION_LIST ${Python3_VERSION})
list(GET PYTHON_VERSION_LIST 0 PYTHON_MAJOR_VERSION)
list(GET PYTHON_VERSION_LIST 1 PYTHON_MINOR_VERSION)
message(STATUS "CPython Version: ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}")
# <<<< Getting Python Major & Minor Version ----------------------------------------------------------

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

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../include/core)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../include/index)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../include/operators)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../include/storage)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/lib/faiss_aarch64/usr/lib/aarch64-linux-gnu
        ${PROJECT_SOURCE_DIR}/lib/SPTAG_aarch64
        ${PROJECT_SOURCE_DIR}/lib/zstd_aarch64
        ${PROJECT_SOURCE_DIR}/lib/libm_aarch64
)

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/lib/zstd_aarch64)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/lib/faiss_aarch64/usr/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../annoy)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../SPTAG/AnnService)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../numpy/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/../fasttext/src)

file(GLOB_RECURSE SRC_FILES
        ${PROJECT_SOURCE_DIR}/../src/core/*.cpp
        ${PROJECT_SOURCE_DIR}/../src/index/*.cpp
        ${PROJECT_SOURCE_DIR}/../src/operators/*.cpp
        ${PROJECT_SOURCE_DIR}/../src/storage/*.cpp
)

add_library(microvecdb SHARED ${SRC_FILES})
#target_link_libraries(microvecdb PUBLIC faiss DistanceUtils SPTAGLibStatic rocksdb pthread m dl zstd)
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})


# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE ${PROJECT_SOURCE_DIR}/../python/pymicrovecdb.cpp)
target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${Python3_LIBRARIES} microvecdb)
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})
