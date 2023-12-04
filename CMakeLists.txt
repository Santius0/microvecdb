cmake_minimum_required(VERSION 3.26)
project(microvecdb)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

include_directories(include annoy faiss/include fasttext/src log4c/include)

link_directories(faiss/lib log4c/lib)

#file(GLOB SOURCES CONFIGURE_DEPENDS "src/*")
#file(GLOB HEADER_SOURCES CONFIGURE_DEPENDS "include/*")

# gathering all the fasttext src files and removing main.cc
file(GLOB FASTTEXT_SOURCES CONFIGURE_DEPENDS "fasttext/src/*.cc")
list(REMOVE_ITEM FASTTEXT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fasttext/src/main.cc")

#add_executable(microvecdb main.cpp ${SOURCES} ${HEADER_SOURCES} ${FASTTEXT_SOURCES})

add_executable(microvecdb main.cpp
        include/kv_store.h
        src/kv_store.cpp
        include/vector_index.h
        src/vector_index.cpp
        include/vector_collection.h
        src/vector_collection.cpp
        include/tests.h
        include/constants.h
        include/fastttext_c.h
        src/fastttext_c.cpp
        ${FASTTEXT_SOURCES}
        include/vectorizer.h
        src/vectorizer.cpp
)

if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
    target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c log4c)
else ()
    target_link_libraries(microvecdb rocksdb faiss faiss_c log4c)
endif()