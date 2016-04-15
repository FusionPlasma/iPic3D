cmake_minimum_required(VERSION 3.4)
project(H5hut_io)

set(HDF5_LIB_PATH "-L/home/vadim/HDF5/lib")
set(MPICH2_LIB_PATH "-L/home/vadim/mpich2-install/lib")
set(H5HUT_LIB_PATH "-L/home/vadim/H5hut/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("/home/vadim/HDF5/include")
include_directories("/home/vadim/mpich2-install/include")
include_directories("/home/vadim/H5hut/include")
link_directories("/home/vadim/HDF5/bin")
link_directories("/home/vadim/HDF5/lib")
link_directories("/home/vadim/mpich2-install/bin")
link_directories("/home/vadim/mpich2-install/lib")
link_directories("/home/vadim/H5hut/lib")
link_libraries(${HDF5_LIB_PATH} -lhdf5 ${HDF5_LIB_PATH} -lhdf5_hl)
link_libraries(H5hut)
link_libraries(${MPICH2_LIB_PATH} -lmpe)

set(SOURCE_FILES
    include/H5hut-io.h
    src/H5hut-io.cpp)

add_executable(H5hut_io ${SOURCE_FILES})