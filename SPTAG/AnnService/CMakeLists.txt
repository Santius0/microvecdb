# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(AnnService ${PROJECT_SOURCE_DIR}/AnnService)
include_directories(${AnnService})

file(GLOB_RECURSE HDR_FILES ${AnnService}/inc/Core/*.h  ${AnnService}/inc/Helper/*.h)
file(GLOB_RECURSE SRC_FILES ${AnnService}/src/Core/*.cpp ${AnnService}/src/Helper/*.cpp)

list(REMOVE_ITEM HDR_FILES
        ${AnnService}/inc/Core/Common/DistanceUtils.h
        ${AnnService}/inc/Core/Common/SIMDUtils.h
        ${AnnService}/inc/Core/Common/InstructionUtils.h
        ${AnnService}/inc/Core/Common/CommonUtils.h
)

list(REMOVE_ITEM SRC_FILES
        ${AnnService}/src/Core/Common/DistanceUtils.cpp
        ${AnnService}/src/Core/Common/SIMDUtils.cpp
        ${AnnService}/src/Core/Common/InstructionUtils.cpp
)

add_library (DistanceUtils STATIC
        inc/Core/Common/DistanceUtils.h
        inc/Core/Common/SIMDUtils.h
        inc/Core/Common/InstructionUtils.h
        inc/Core/Common/CommonUtils.h
        src/Core/Common/DistanceUtils.cpp
        src/Core/Common/SIMDUtils.cpp
        src/Core/Common/InstructionUtils.cpp
)

#if(CROSS_COMPILE)
#    target_compile_options(DistanceUtils PRIVATE -mfpu=neon -mfloat-abi=softfp -fPIC) # armv7-a
    target_compile_options(DistanceUtils PRIVATE -march=armv8-a -fPIC)  # NEON is enabled by default in ARMv8
#else()
#    target_compile_options(DistanceUtils PRIVATE -mavx2 -mavx -msse -msse2 -mavx512f -mavx512bw -mavx512dq -fPIC)
#endif()

# if an arm device, use precompiled libs other use libs compiled from source
if(CROSS_COMPILE)
    message(STATUS "Using prebuilt aarch64 zstd")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../zstd_aarch64)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../zstd_aarch64)
    set(ZSTD_LIB_SHARED "zstd")
    set(ZSTD_LIB_STATIC "zstd")
else ()
    set(Zstd ${PROJECT_SOURCE_DIR}/ThirdParty/zstd)
    include_directories(${Zstd}/lib)
    set(ZSTD_LIB_SHARED "libzstd_shared")
    set(ZSTD_LIB_STATIC "libzstd_static")
endif ()

message(STATUS "ZSTD_LIB_SHARED: ${ZSTD_LIB_SHARED}")
message(STATUS "ZSTD_LIB_STATIC: ${ZSTD_LIB_STATIC}")

#add_library (SPTAGLib SHARED ${SRC_FILES} ${HDR_FILES})
#target_link_libraries (SPTAGLib DistanceUtils ${ZSTD_LIB_SHARED} ${NUMA_LIBRARY})
#if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
#    target_compile_options(SPTAGLib PRIVATE -fPIC)
#endif()

add_library (SPTAGLibStatic STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries (SPTAGLibStatic DistanceUtils zstd ${NUMA_LIBRARY_STATIC})
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(SPTAGLibStatic PRIVATE -fPIC)
endif()

install(TARGETS SPTAGLibStatic
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

if (NOT LIBRARYONLY)
    file(GLOB BUILDER_FILES ${AnnService}/src/IndexBuilder/*.cpp)
    add_executable (indexbuilder ${BUILDER_FILES})
    target_link_libraries(indexbuilder ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB SEARCHER_FILES ${AnnService}/src/IndexSearcher/*.cpp)
    add_executable (indexsearcher ${SEARCHER_FILES})
    target_link_libraries(indexsearcher ${Boost_LIBRARIES} SPTAGLibStatic)

    file(GLOB QUANTIZER_HDR_FILES ${AnnService}/inc/Quantizer/*.h)
    file(GLOB QUANTIZER_FILES ${AnnService}/src/Quantizer/*.cpp)
    add_executable (quantizer ${QUANTIZER_FILES} ${QUANTIZER_HDR_FILES})
    target_link_libraries(quantizer ${Boost_LIBRARIES} SPTAGLibStatic)

    install(TARGETS indexbuilder indexsearcher quantizer
            RUNTIME DESTINATION bin
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib)
endif()

file(GLOB_RECURSE SSD_SERVING_HDR_FILES ${AnnService}/inc/SSDServing/*.h)
file(GLOB_RECURSE SSD_SERVING_FILES ${AnnService}/src/SSDServing/*.cpp)

add_executable(ssdserving ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdserving SPTAGLibStatic ${Boost_LIBRARIES})
target_compile_definitions(ssdserving PRIVATE _exe)

# for Test
add_library(ssdservingLib ${SSD_SERVING_HDR_FILES} ${SSD_SERVING_FILES})
target_link_libraries(ssdservingLib SPTAGLibStatic ${Boost_LIBRARIES})

find_package(MPI)
if (MPI_FOUND)
    message (STATUS "Found MPI.")
    message (STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")
    message (STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
    include_directories (${MPI_CXX_INCLUDE_PATH})
    file(GLOB PARTITION_FILES ${AnnService}/src/BalancedDataPartition/*.cpp)
    add_executable(balanceddatapartition ${PARTITION_FILES})
    target_link_libraries(balanceddatapartition SPTAGLibStatic ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

install(TARGETS ssdservingLib ssdserving
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
