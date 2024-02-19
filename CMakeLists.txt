cmake_minimum_required(VERSION 3.26)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-std=c++17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

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
find_package(OpenMP REQUIRED)
find_package(PythonLibs REQUIRED)
string(REPLACE "." ";" PYTHON_VERSION_LIST ${PYTHONLIBS_VERSION_STRING})
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

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/core)   # all targets should have access to core headers
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/index)   # all targets should have access to core headers
link_directories(lib)                                                     # all targets should be able to link against the contents of the lib directory

#include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/faiss/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/numpy/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/fasttext/src)

# <<<< Configuring sources headers and link directories ----------------------------------------------------------

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# Configuring targets >>>> ----------------------------------------------------------
# microvecdb_core:
add_library(microvecdb_core SHARED
        include/core/utils.h
        include/core/quantization.h
        include/core/preprocess.h

        src/core/utils.cpp
        src/core/quantization.cpp
        src/core/preprocess.cpp
        include/core/server.h
        src/core/server.cpp
        include/core/settings.h
        include/index/heap.h
        include/core/wal_old.h
        include/core/wal.h
        src/core/wal.cpp
)
target_link_libraries(microvecdb_core PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C zmq pthread spdlog::spdlog)
set_target_properties(microvecdb_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb_index:
add_library(microvecdb_index SHARED
        include/index/index.h
#        include/index/faiss_flat_index.h
        include/index/distances.h
#        src/index/faiss_flat_index.cpp
        include/index/flat_index.h
        src/index/flat_index.cpp
)
target_include_directories(microvecdb_index PUBLIC include/index)
#target_link_directories(microvecdb_index PUBLIC faiss/lib)
target_link_libraries(microvecdb_index PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C openblas microvecdb_core
#        faiss faiss_c
)
set_target_properties(microvecdb_index PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb_storage:
add_library(microvecdb_storage SHARED
        src/storage/storage.cpp
        include/storage/storage.h
)
target_include_directories(microvecdb_storage PUBLIC include/storage)
target_link_libraries(microvecdb_storage PUBLIC microvecdb_core rocksdb)
set_target_properties(microvecdb_storage PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb_vectorizer
file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc
add_library(microvecdb_vectorizer SHARED
        src/vectorizer/vectorizer.cpp
        ${FASTTEXT_SOURCES}
)
target_include_directories(microvecdb_vectorizer PUBLIC include/vectorizer)
target_link_libraries(microvecdb_vectorizer PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C microvecdb_core)
set_target_properties(microvecdb_vectorizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb:
add_library(microvecdb SHARED src/db.cpp)
target_include_directories(microvecdb PUBLIC include)
target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C microvecdb_core microvecdb_index microvecdb_storage microvecdb_vectorizer)
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# microvecdb_cli:
FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.3.1
)
FetchContent_MakeAvailable(cli11)
add_executable(microvecdb_cli src/cli.cpp)
target_include_directories(microvecdb_cli PUBLIC include)
target_link_libraries(microvecdb_cli PRIVATE CLI11::CLI11)
target_link_libraries(microvecdb_cli PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C microvecdb microvecdb_core microvecdb_index microvecdb_storage microvecdb_vectorizer)
set_target_properties(microvecdb_cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE
        python/pymicrovecdb.cpp
)
target_include_directories(${PYTHON_EXTENSION_TARGET} PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_directories(${PYTHON_EXTENSION_TARGET} PUBLIC lib)
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${PYTHON_LIBRARIES} microvecdb)
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(PYTHON_PACKAGE_BUILD_SCRIPT "/python/pymicrovecdb.sh")
#file(CHMOD +x ${PYTHON_PACKAGE_BUILD_SCRIPT})
#add_custom_command(
#        TARGET ${PYTHON_EXTENSION_TARGET} POST_BUILD
#        COMMAND ${PYTHON_PACKAGE_BUILD_SCRIPT}
#        COMMENT "Building python package"
#)

# main:
add_executable(microvecdb_main main.cpp)
target_link_libraries(microvecdb_main PUBLIC microvecdb zmq pthread Eigen3::Eigen)

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


FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)