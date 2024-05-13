cmake_minimum_required(VERSION 3.27)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fPIC --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE) # neon is included by default in the armv8-a architecture
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC --sysroot=${CMAKE_SYSROOT} -march=armv8-a -O3" CACHE STRING "" FORCE)


add_definitions(-DANNOYLIB_MULTITHREADED_BUILD)

# Disable warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-w) # Turn off all warnings for GCC/Clang
elseif(MSVC)
    add_compile_options(/W0) # Turn off all warnings for MSVC
endif()

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(CUDAToolkit REQUIRED)
#find_package(OpenMP REQUIRED)
#include(FetchContent)

set(LIB_DIR ${PROJECT_SOURCE_DIR}/lib)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build type: Release")
#    set(LIB_DIR ${PROJECT_SOURCE_DIR}/release/lib)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Build type: Debug")
#    set(LIB_DIR ${PROJECT_SOURCE_DIR}/debug/lib)
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()

## making sure output directory exists
file(MAKE_DIRECTORY "${LIB_DIR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


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
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
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

#link_directories(
#        ${LIB_DIR}/lib
#        ${PROJECT_SOURCE_DIR}/faiss/lib
#        ${PROJECT_SOURCE_DIR}/SPTAG/Release
#)

link_directories(
        ${LIB_DIR}/lib
        ${PROJECT_SOURCE_DIR}/jetson/lib/faiss_aarch64/usr/lib/aarch64-linux-gnu
        ${PROJECT_SOURCE_DIR}/jetson/lib/SPTAG_aarch64
)

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
add_library(microvecdb SHARED ${SRC_FILES})
#target_include_directories(microvecdb PUBLIC "${MKL_LIBRARY_DIR}")
target_link_libraries(microvecdb PUBLIC
#        OpenMP::OpenMP_CXX OpenMP::OpenMP_C
#        ${MKL_INTEL_LP64} ${MKL_CORE} ${MKL_SEQUENTIAL}
        faiss
        DistanceUtils
        SPTAGLibStatic
        rocksdb
        pthread m dl
        zstd
)
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})

# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE python/pymicrovecdb.cpp)
target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${Python3_LIBRARIES} microvecdb)
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_DIR})

# main:
add_executable(microvecdb_main main.cpp)
target_link_libraries(microvecdb_main PUBLIC microvecdb DistanceUtils SPTAGLibStatic pthread)

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