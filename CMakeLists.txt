cmake_minimum_required(VERSION 3.26)
project(microvecdb)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP)

include_directories(include annoy faiss/include
#        faiss_env/include
)

link_directories(
        faiss/lib
#        faiss_env/lib
)

#file(GLOB SOURCES CONFIGURE_DEPENDS "src/*")
#add_executable(microvecdb main.cpp ${SOURCES})

add_executable(microvecdb main.c
        include/kv_store.h
        src/kv_store.c
        include/vector_index.h
        src/vector_index.c
        include/vector_collection.h
        src/vector_collection.c
        include/tests.h
        include/constants.h
        include/utils.h
        src/utils.c
)

if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
    target_link_libraries(microvecdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)
else ()
    target_link_libraries(microvecdb rocksdb faiss faiss_c)
endif()