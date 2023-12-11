cmake_minimum_required(VERSION 3.26)
project(microvecdb)

#message("")
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(PythonLibs REQUIRED)

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

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/faiss_src)

link_directories(lib faiss/lib ${CMAKE_BINARY_DIR})

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "include/*")
file(GLOB_RECURSE FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc

#add_library(microindex SHARED ${SOURCES})
add_library(microvecdb SHARED ${SOURCES} ${HEADERS} ${FASTTEXT_SOURCES})
add_library(microvecdb_py MODULE python/microvecdb_py.cpp)
add_executable(microvecdb_main main.cpp)
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

if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
#    target_link_libraries(microindex PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
    target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)
else ()
#    target_link_libraries(microindex)
    target_link_libraries(microvecdb PUBLIC rocksdb faiss faiss_c)
endif()
target_link_libraries(microvecdb_main PUBLIC microvecdb)
target_link_libraries(microvecdb_py ${PYTHON_LIBRARIES})

set_target_properties(microvecdb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
