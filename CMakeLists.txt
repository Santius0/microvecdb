cmake_minimum_required(VERSION 3.26)
project(confluxdb)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

include_directories(include annoy faiss/include
#        faiss_env/include
)

link_directories(
        faiss/lib
#        faiss_env/lib
)

#file(GLOB SOURCES CONFIGURE_DEPENDS "src/*")
#add_executable(confluxdb main.cpp ${SOURCES})

add_executable(confluxdb main.c
        include/kv_store.h
        src/kv_store.c
        include/vector_index.h
        src/vector_index.c
        include/vector_db.h
        src/vector_db.c
)

if(OpenMP_CXX_FOUND OR OpenMP_C_FOUND)
    target_link_libraries(confluxdb PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C rocksdb faiss faiss_c)
else ()
    target_link_libraries(confluxdb rocksdb faiss faiss_c)
endif()