cmake_minimum_required(VERSION 3.27)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)
#add_compile_options(-std=c++17)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type: Release")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/release/lib)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type: Debug")
    set(LIB_DIR ${PROJECT_SOURCE_DIR}/debug/lib)
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()

#find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")

#include(FetchContent)

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#find_program(READELF_EXECUTABLE readelf)
#if(NOT READELF_EXECUTABLE)
#    message(FATAL_ERROR "readelf not found")
#endif()
#
#find_program(SIZE_EXECUTABLE size)
#if(NOT SIZE_EXECUTABLE)
#    message(FATAL_ERROR "size tool not found")
#endif()

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
#find_package(PythonLibs REQUIRED) # deprecated. use find_package(Python3 COMPONENTS Interpreter Development REQUIRED) for cmake 3.27.x +
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#string(REPLACE "." ";" PYTHON_VERSION_LIST ${PYTHONLIBS_VERSION_STRING}) # deprecated with new find_package method use Python3_VERSION instead
string(REPLACE "." ";" PYTHON_VERSION_LIST ${Python3_VERSION})
list(GET PYTHON_VERSION_LIST 0 PYTHON_MAJOR_VERSION)
list(GET PYTHON_VERSION_LIST 1 PYTHON_MINOR_VERSION)
message(STATUS "CPython Version: ${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}")
# <<<< Getting Python Major & Minor Version ----------------------------------------------------------

# Configuring sources headers and link directories >>>> ----------------------------------------------------------
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

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/core)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/index)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/operators)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/storage)

link_directories(
        ${LIB_DIR}/lib
        ${PROJECT_SOURCE_DIR}/faiss/lib
#        ${PROJECT_SOURCE_DIR}/DiskANN/build/src
)

#include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/DiskANN/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/faiss/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/numpy/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/fasttext/src)

# <<<< Configuring sources headers and link directories ----------------------------------------------------------

#FetchContent_Declare(
#        spdlog
#        GIT_REPOSITORY https://github.com/gabime/spdlog.git
#        GIT_TAG v1.13.0
#)
#FetchContent_MakeAvailable(spdlog)

# Configuring targets >>>> ----------------------------------------------------------
## microvecdb_core:
#add_library(microvecdb_core SHARED
#        src/db.cpp
#        src/core/utils.cpp
#        src/operators/operators.cpp
#)
#target_link_libraries(microvecdb_core PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C
##        zmq
#        pthread
##        spdlog::spdlog
##        Eigen3::Eigen
#)
#set_target_properties(microvecdb_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Find MKL libraries
#set(MKL_LIBRARY_DIR /opt/intel/oneapi/mkl/latest/lib/intel64)
#find_library(MKL_CORE libmkl_core.so PATHS ${MKL_LIBRARY_DIR} NO_DEFAULT_PATH)
#find_library(MKL_INTEL_LP64 libmkl_intel_lp64.so PATHS ${MKL_LIBRARY_DIR} NO_DEFAULT_PATH)
#find_library(MKL_SEQUENTIAL libmkl_sequential.so PATHS ${MKL_LIBRARY_DIR} NO_DEFAULT_PATH)


## microvecdb_index:
#add_library(microvecdb_index SHARED
#        src/index/distances.cpp
#        src/index/knn.cpp
#        src/index/flat_index.cpp
#        src/index/faiss_flat_index.cpp
#)
#
#target_include_directories(microvecdb_index PUBLIC include/index "${MKL_LIBRARY_DIR}")
#target_link_directories(microvecdb_index PUBLIC faiss/lib DiskANN/build/src)
#target_link_libraries(microvecdb_index PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C
##        ${MKL_INTEL_LP64} ${MKL_CORE} ${MKL_SEQUENTIAL}
#        pthread m dl
##        openblas
#        microvecdb_core
#        faiss faiss_c
##        diskann diskann_s
#)
#set_target_properties(microvecdb_index PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


## microvecdb_storage:
#add_library(microvecdb_storage SHARED
#        src/storage/storage.cpp
#)
#target_include_directories(microvecdb_storage PUBLIC include/storage)
#target_link_libraries(microvecdb_storage PUBLIC microvecdb_core rocksdb)
#set_target_properties(microvecdb_storage PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb_vectorizer
#file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
#list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc
#add_library(microvecdb_vectorizer SHARED
#        src/vectorizer/vectorizer.cpp
#        ${FASTTEXT_SOURCES}
#)
#target_include_directories(microvecdb_vectorizer PUBLIC include/vectorizer)
#target_link_libraries(microvecdb_vectorizer PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C microvecdb_core)
#set_target_properties(microvecdb_vectorizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


file(GLOB_RECURSE SRC_FILES
        src/core/*.cpp
        src/index/*.cpp
        src/operators/*.cpp
        src/storage/*.cpp
)

# microvecdb:
add_library(microvecdb SHARED ${SRC_FILES})
#target_include_directories(microvecdb PUBLIC "${MKL_LIBRARY_DIR}")
target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C
#        ${MKL_INTEL_LP64} ${MKL_CORE} ${MKL_SEQUENTIAL}
        pthread m dl faiss faiss_c
#        diskann diskann_s
        rocksdb
)
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})


## microvecdb_cli:
#FetchContent_Declare(
#        cli11
#        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
#        GIT_TAG v2.3.1
#)

#FetchContent_MakeAvailable(cli11)
#add_executable(microvecdb_cli src/cli.cpp)
#target_include_directories(microvecdb_cli PUBLIC include)
#target_link_libraries(microvecdb_cli PRIVATE CLI11::CLI11)
#target_link_libraries(microvecdb_cli PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C microvecdb microvecdb_core microvecdb_index microvecdb_storage microvecdb_vectorizer)
#set_target_properties(microvecdb_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE
        python/pymicrovecdb.cpp
)
#target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${Python3_INCLUDE_DIRS})
target_link_directories(${PYTHON_EXTENSION_TARGET} PUBLIC lib)
#target_link_libraries(${PYTHON_EXTENSION_TARGET} ${PYTHON_LIBRARIES} microvecdb)
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${Python3_LIBRARIES} microvecdb)
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# main:
add_executable(microvecdb_main main.cpp)
target_link_libraries(microvecdb_main PUBLIC microvecdb pthread)

#################################################
# Check if we are cross-compiling for Jetson
##if(JETSON_CROSS_COMPILE)
#
#    # Jetson Nano-specific executable, using Linaro for cross-compilation
#    add_executable(my_jetson_app main.cpp)
#
#    # Set custom compiler for my_jetson_app
#    set_target_properties(my_jetson_app PROPERTIES
#            CXX_STANDARD 17
#            CXX_STANDARD_REQUIRED YES
#            CXX_EXTENSIONS NO
#    )
#    # Specify the path to your Linaro toolchain here
#    set(LINARO_TOOLCHAIN_PATH "/usr/bin")
#
#    set_property(TARGET my_jetson_app PROPERTY CMAKE_SYSTEM_NAME Linux)
#    set_property(TARGET my_jetson_app PROPERTY CMAKE_C_COMPILER "${LINARO_TOOLCHAIN_PATH}/arm-linux-gnueabihf-gcc")
#    set_property(TARGET my_jetson_app PROPERTY CMAKE_CXX_COMPILER "${LINARO_TOOLCHAIN_PATH}/arm-linux-gnueabihf-g++")
#    target_compile_options(my_jetson_app PRIVATE -march=armv8-a)


    # Manually specify the cross-compiler from Linaro


    # Target-specific compile options for Jetson Nano
#    target_include_directories(my_jetson_app PRIVATE "${LINARO_TOOLCHAIN_PATH}/include")
#    target_link_libraries(my_jetson_app PRIVATE )

    # Additional configurations specific to the Jetson target
    # For example, setting the SYSROOT (if needed)
    # set(CMAKE_SYSROOT "${LINARO_TOOLCHAIN_PATH}/arm-linux-gnueabihf/libc")
#endif()

# <<<< Configuring targets ----------------------------------------------------------

## making sure output directory exists
file(MAKE_DIRECTORY "${LIBRARY_OUTPUT_DIRECTORY}")
#
## Post build analysis >>>> ----------------------------------------------------------
##set(SIZE_OUTPUT_FILE "${CMAKE_BINARY_DIR}/size_output.txt")
##set(READELF_OUTPUT_FILE "${CMAKE_BINARY_DIR}/readelf_output.txt")
##add_custom_command(
##        TARGET microindex POST_BUILD
##        COMMAND ${SIZE_EXECUTABLE} $<TARGET_FILE:microindex> > ${SIZE_OUTPUT_FILE}
##        COMMENT "Running size on microindex"
##        COMMAND ${READELF_EXECUTABLE} -s $<TARGET_FILE:microindex> | sort -n -k3 > ${READELF_OUTPUT_FILE}
##        COMMENT "Running readelf on microindex"
##)
## <<<< Post build analysis ----------------------------------------------------------


#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
#)
## For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

#enable_testing()
#add_subdirectory(tests)
#add_subdirectory(benchmarks)