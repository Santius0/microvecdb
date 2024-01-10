cmake_minimum_required(VERSION 3.26)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)

find_program(READELF_EXECUTABLE readelf)
if(NOT READELF_EXECUTABLE)
    message(FATAL_ERROR "readelf not found")
endif()

find_program(SIZE_EXECUTABLE size)
if(NOT SIZE_EXECUTABLE)
    message(FATAL_ERROR "size tool not found")
endif()

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

include_directories(${PYTHON_INCLUDE_DIRS})
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/faiss/include)
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/numpy/include)
#include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/fasttext/src)
#include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/rocksdb/include)
#
#link_directories(lib faiss/lib ${CMAKE_BINARY_DIR})

#file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*")
#file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "include/*")
#file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
#list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc
# <<<< Configuring sources headers and link directories ----------------------------------------------------------

# Configuring targets >>>> ----------------------------------------------------------
# microindex:
add_library(microindex SHARED
        src/index/index.cpp
        src/index/faiss_flat_index.cpp
        include/preprocess.h
        include/quantization.h
        include/index/index.h
        #        include/index/faiss_flat_index.h
#        include/index/faiss_ivf_index.h
#        include/index/faiss_hnsw_index.h
#        include/index/faiss_pq_index.h
)

target_include_directories(microindex PUBLIC include/index)
target_link_directories(microindex PUBLIC faiss/lib)
target_link_libraries(microindex PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C faiss faiss_c)
set_target_properties(microindex PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)


# storage:
# TODO: implement and configure rocksdb stuff
#add_library(microstorage PUBLIC <insert sources here>)
#target_link_directories(microstorage PUBLIC rocksdb/lib)

# microvecdb:
# add_library(microvecdb SHARED ${SOURCES} ${FASTTEXT_SOURCES})
# target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)
#set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# microvecdb.cpython
set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE python/pymicrovecdb.cpp
        include/index/faiss_flat_index.h)
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${PYTHON_LIBRARIES} microindex)
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
target_link_libraries(microvecdb_main PUBLIC microindex zmq pthread)
# <<<< Configuring targets ----------------------------------------------------------

# making sure output directory exists
file(MAKE_DIRECTORY "${LIBRARY_OUTPUT_DIRECTORY}")

# Post build analysis >>>> ----------------------------------------------------------
set(SIZE_OUTPUT_FILE "${CMAKE_BINARY_DIR}/size_output.txt")
set(READELF_OUTPUT_FILE "${CMAKE_BINARY_DIR}/readelf_output.txt")
add_custom_command(
        TARGET microindex POST_BUILD
        COMMAND ${SIZE_EXECUTABLE} $<TARGET_FILE:microindex> > ${SIZE_OUTPUT_FILE}
        COMMENT "Running size on microindex"
        COMMAND ${READELF_EXECUTABLE} -s $<TARGET_FILE:microindex> | sort -n -k3 > ${READELF_OUTPUT_FILE}
        COMMENT "Running readelf on microindex"
)
# <<<< Post build analysis ----------------------------------------------------------


include(FetchContent)
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