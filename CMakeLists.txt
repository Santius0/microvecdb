cmake_minimum_required(VERSION 3.26)
project(microvecdb)

# Constructing Platform Tag >>>> ----------------------------------------------------------
set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

# Detect the operating system
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS "linux-gnu")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(OS "win32")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    set(OS "darwin")
else()
    set(OS ${CMAKE_SYSTEM_NAME})
endif()

# Construct the platform tag
set(PLATFORM_TAG "${ARCHITECTURE}-${OS}")

# <<<< Constructing Platform Tag  ----------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(PythonLibs REQUIRED)
string(REPLACE "." ";" PYTHON_VERSION_LIST ${PYTHONLIBS_VERSION_STRING})
list(GET PYTHON_VERSION_LIST 0 PYTHON_MAJOR_VERSION)
list(GET PYTHON_VERSION_LIST 1 PYTHON_MINOR_VERSION)

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

add_library(microvecdb SHARED ${SOURCES} ${HEADERS} ${FASTTEXT_SOURCES})
add_executable(microvecdb_main main.cpp)

set(python_extension_target "microvecdb.cpython-${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-${PLATFORM_TAG}")
add_library(${python_extension_target} MODULE python/pymicrovecdb.cpp)

#add_custom_target(
#        BuildPythonPackage
#        COMMAND ${PYTHON_EXECUTABLE} python/setup.py build
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#)
#
#add_custom_command(
#        TARGET BuildPythonPackage
#        POST_BUILD
#        COMMAND ${PYTHON_EXECUTABLE} python/setup.py bdist_wheel
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#        COMMENT "Building Python Package"
#)

#set_target_properties(microvecdb_py PROPERTIES PREFIX "")

#if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
#    target_link_libraries(microindex PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
    target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)
#else ()
#    target_link_libraries(microindex)
#    target_link_libraries(microvecdb PUBLIC rocksdb faiss faiss_c)
#endif()
target_link_libraries(microvecdb_main PUBLIC microvecdb)
target_link_libraries(${python_extension_target} ${PYTHON_LIBRARIES} microvecdb)


set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set_target_properties(${python_extension_target} PROPERTIES PREFIX "")
set_target_properties(${python_extension_target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY "${LIBRARY_OUTPUT_DIRECTORY}")