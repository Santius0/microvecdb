cmake_minimum_required(VERSION 3.27)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fPIC -march=armv8-a -O3" CACHE STRING "" FORCE) # neon is included by default in the armv8-a architecture
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -march=armv8-a -O3" CACHE STRING "" FORCE)
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -O3 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -O3 -fPIC")
    find_package(OpenMP REQUIRED)
endif ()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic")

add_definitions(-DANNOYLIB_MULTITHREADED_BUILD) # enable annoy multithreaded build

# Disable warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-w)
elseif(MSVC)
    add_compile_options(/W0)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE release CACHE STRING "Build types: release debug" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type: Release")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type: Debug")
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()

set(LIB_DIR "${PROJECT_SOURCE_DIR}/lib/")
set(BIN_DIR "${PROJECT_SOURCE_DIR}/bin/")

# making sure output directories exists
file(MAKE_DIRECTORY "${LIB_DIR}")
file(MAKE_DIRECTORY "${BIN_DIR}")

set(LIBRARY_OUTPUT_PATH ${LIB_DIR})
set(EXECUTABLE_OUTPUT_PATH ${BIN_DIR})

message(STATUS "LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")
message(STATUS "EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH}")


# Constructing Platform Tag >>>> ----------------------------------------------------------
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS "linux-gnu")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(OS "win32")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(OS "darwin")
else()
    set(OS ${CMAKE_SYSTEM_NAME})
endif()
set(PLATFORM_TAG "${CMAKE_SYSTEM_PROCESSOR}-${OS}")
message(STATUS "Platform Tag: ${PLATFORM_TAG}")
# <<<< Constructing Platform Tag  ----------------------------------------------------------

# Getting Python Major & Minor Version >>>> ----------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
string(REPLACE "." ";" PYTHON_VERSION_LIST ${Python3_VERSION})
list(GET PYTHON_VERSION_LIST 0 PYTHON_MAJOR_VERSION)
list(GET PYTHON_VERSION_LIST 1 PYTHON_MINOR_VERSION)
message(STATUS "CPython Version: ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}")
# <<<< Getting Python Major & Minor Version ----------------------------------------------------------

# Configuring sources headers and link directories >>>> ----------------------------------------------------------
function(include_sub_directories_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
#        message("Including directory: ${root_dir}")
        include_directories(${root_dir})
        file(GLOB sub_dirs LIST_DIRECTORIES true RELATIVE ${root_dir} ${root_dir}/*)
        foreach(sub_dir ${sub_dirs})
            if (IS_DIRECTORY ${root_dir}/${sub_dir})
                include_sub_directories_recursively(${root_dir}/${sub_dir})
            endif()
        endforeach()
    endif()
endfunction()

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/core)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/index)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/operators)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/storage)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    link_directories(
            ${LIB_DIR}/lib
            ${PROJECT_SOURCE_DIR}/faiss/lib/aarch64
            ${PROJECT_SOURCE_DIR}/SPTAG/Release/arm/lib
    )
else ()
    link_directories(
            ${LIB_DIR}/lib
            ${PROJECT_SOURCE_DIR}/faiss/lib
            ${PROJECT_SOURCE_DIR}/SPTAG/Release/x86_64/lib
    )
endif ()

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/faiss/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/annoy)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/SPTAG/AnnService)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/numpy/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/fasttext/src)

# <<<< Configuring sources headers and link directories ----------------------------------------------------------

file(GLOB_RECURSE SRC_FILES
        src/core/*.cpp
        src/index/*.cpp
        src/operators/*.cpp
        src/storage/*.cpp
)


# microvecdb:
set(MICROVECDB_SHARED_LIB "microvecdb_${CMAKE_SYSTEM_PROCESSOR}")
add_library(${MICROVECDB_SHARED_LIB} SHARED ${SRC_FILES})
#target_include_directories(microvecdb PUBLIC "${MKL_LIBRARY_DIR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_libraries(${MICROVECDB_SHARED_LIB} PUBLIC
        faiss
        DistanceUtils
#        SPTAGLibStatic
        rocksdb
        pthread m dl
        zstd
        openblas
        stdc++fs
    )
else ()
    target_link_libraries(${MICROVECDB_SHARED_LIB} PUBLIC
            OpenMP::OpenMP_CXX OpenMP::OpenMP_C
            # ${MKL_INTEL_LP64} ${MKL_CORE} ${MKL_SEQUENTIAL}
            faiss
            DistanceUtils
#            SPTAGLibStatic
            rocksdb
            pthread m dl
            zstd
            openblas
            stdc++fs
    )
endif ()

# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE python/pymicrovecdb.cpp)
target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${Python3_LIBRARIES} ${MICROVECDB_SHARED_LIB})
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")

# main:
#if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"))
add_executable(microvecdb_main main.cpp)
target_link_libraries(microvecdb_main PUBLIC ${MICROVECDB_SHARED_LIB} DistanceUtils SPTAGLibStatic pthread)
#endif ()
# <<<< Configuring targets ----------------------------------------------------------


# Post build analysis >>>> ----------------------------------------------------------
#find_program(READELF_EXECUTABLE readelf)
#if(READELF_EXECUTABLE)
#    message(STATUS "readelf found")
#else ()
#    message(STATUS "readelf not found")
#endif()
#
#find_program(SIZE_EXECUTABLE size)
#if(SIZE_EXECUTABLE)
#    message(STATUS "size tool found")
#else ()
#    message(STATUS "size tool not found")
#endif()
#
#if(READELF_EXECUTABLE AND SIZE_EXECUTABLE)
#    set(SIZE_OUTPUT_FILE "${LIB_DIR}/../size_output.txt")
#    set(READELF_OUTPUT_FILE "${LIB_DIR}/../readelf_output.txt")
#    add_custom_command(
#            TARGET microvecdb POST_BUILD
#            COMMAND ${SIZE_EXECUTABLE} $<TARGET_FILE:microvecdb> > ${SIZE_OUTPUT_FILE}
#            COMMENT "Running size on microvecdb..."
#            COMMAND ${READELF_EXECUTABLE} -s $<TARGET_FILE:microvecdb> | sort -n -k3 > ${READELF_OUTPUT_FILE}
#            COMMENT "Running readelf on microvecdb..."
#)
#endif ()
# <<<< Post build analysis ----------------------------------------------------------

#add_subdirectory(benchmarks)