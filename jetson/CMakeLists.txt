cmake_minimum_required(VERSION 3.26)
project(microvecdb-jetson-nano-2gb)

#message(FATAL_ERROR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
#set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_C_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
#set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/linario/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")
set(CMAKE_C_FLAGS "-march=armv8-a")
#set(CMAKE_C_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=neon")
#set(CMAKE_C_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8")
#set(CMAKE_CXX_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=neon")
set(CMAKE_CXX_FLAGS "-march=armv8-a")
#set(CMAKE_CXX_FLAGS "-march=armv8-a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8")
set(CMAKE_CXX_STANDARD 17)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/src)

add_executable(my_jetson_app ../jetson_main_test.cpp)
#target_link_directories(my_jetson_app PRIVATE Eigen3::Eigen)

#target_compile_options(my_jetson_app PRIVATE -march=armv8-a)