cmake_minimum_required(VERSION 3.26)
project(microvecdb)
set(CMAKE_CXX_STANDARD 17)

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
include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/fasttext/src)
#include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/rocksdb/include)

link_directories(lib rocksdb/lib faiss/lib ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE MICROINDEX_SOURCES CONFIGURE_DEPENDS "src/index/*.cpp")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "include/*")
file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc
# <<<< Configuring sources headers and link directories ----------------------------------------------------------

# Specifying targets >>>> ----------------------------------------------------------
add_library(microvecdb SHARED ${SOURCES} ${HEADERS} ${FASTTEXT_SOURCES})
add_executable(microvecdb_main main.cpp)

set(PYTHON_EXTENSION_TARGET "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${PYTHON_EXTENSION_TARGET} MODULE python/pymicrovecdb.cpp)
# <<<< Specifying targets ----------------------------------------------------------

# Setting Up link libraries targets >>>> ----------------------------------------------------------
#target_link_libraries(microindex PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)

target_link_libraries(microvecdb_main PUBLIC microvecdb)
target_link_libraries(${PYTHON_EXTENSION_TARGET} ${PYTHON_LIBRARIES} microvecdb)
# <<<< Setting Up link libraries targets ----------------------------------------------------------

# Setting up additional properties >>>> ----------------------------------------------------------
set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES PREFIX "")
set_target_properties(${PYTHON_EXTENSION_TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY "${LIBRARY_OUTPUT_DIRECTORY}")
# <<<< Setting up additional properties ----------------------------------------------------------
