cmake_minimum_required(VERSION 3.26)
project(confluxdb)

set(CMAKE_CXX_STANDARD 11)

#file(GLOB MKL_LIBRARIES CONFIGURE_DEPENDS "mkl/mkl/*.so")

include_directories(
        include
        annoy
#        faiss
        faiss_env/include
)

link_directories(
#        faiss/lib
        faiss_env/lib
)

add_executable(confluxdb main.c
        src/confluxstorage.cpp
        include/confluxstorage.h
        include/confluxindex.h
        src/confluxindex.cpp
        include/kv_store.h
        src/kv_store.c
        include/vector_db.h
)

target_link_libraries(
        confluxdb
        rocksdb
        faiss
#        ${MKL_LIBRARIES}
)
