cmake_minimum_required(VERSION 3.26)
project(microvecdb)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include faiss/include fasttext/src ${PYTHON_INCLUDE_PATH})

link_directories(faiss/lib ${CMAKE_BINARY_DIR})

file(GLOB SOURCES CONFIGURE_DEPENDS "src/*")
file(GLOB HEADERS CONFIGURE_DEPENDS "include/*")
file(GLOB FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc") # gathering all the fasttext src files and removing main.cc

add_library(microvecdb SHARED ${SOURCES} ${HEADERS} ${FASTTEXT_SOURCES})
add_executable(microvecdb_main main.cpp)

if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
    target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c ${PYTHON_LIBRARIES})
else ()
    target_link_libraries(microvecdb PUBLIC rocksdb faiss faiss_c ${PYTHON_LIBRARIES})
endif()
target_link_libraries(microvecdb_main PUBLIC microvecdb ${PYTHON_LIBRARIES})
