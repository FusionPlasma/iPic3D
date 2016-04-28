cmake_minimum_required(VERSION 3.4)
project(iPic3D)

set(HDF5_LIB_PATH "-L/home/vadim/HDF5/lib")
set(MPICH2_LIB_PATH "-L/home/vadim/mpich2-install/lib")
set(H5HUT_LIB_PATH "-L/home/vadim/H5hut/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_COMPILER "/home/vadim/mpich2-install/bin/mpicxx")
set(CMAKE_C_COMPILER "/home/vadim/mpich2-install/bin/mpicc")
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
    bc/BcFields3D.cpp
    bc/BcParticles.cpp
    communication/ComBasic3D.cpp
    communication/ComInterpNodes3D.cpp
    communication/ComNodes3D.cpp
    communication/ComParser3D.cpp
    communication/ComParticles3D.cpp
    communication/VCtopology3D.cpp
    ConfigFile/src/ConfigFile.cpp
    ConfigFile/src/ConfigFile.h
    fields/BCStructure.cpp
    fields/EMfields3D.cpp
    grids/Grid3DCU.cpp
    include/Alloc.h
    include/asserts.h
    include/Basic.h
    include/BcFields3D.h
    include/BcParticles.h
    include/BCStructure.h
    include/Bessel.h
    include/CG.h
    include/Collective.h
    include/ComBasic3D.h
    include/ComInterpNodes3D.h
    include/ComNodes3D.h
    include/ComParser3D.h
    include/ComParticles3D.h
    include/ConfigFile.h
    include/debug.h
    include/EllipticF.h
    include/EMfields3D.h
    include/Field.h
    include/GMRES.h
    include/Grid.h
    include/Grid3DCU.h
    include/input_array.h
    include/InterfaceFluid.h
    include/iPic3D.h
    include/MPIdata.h
    include/ParallelIO.h
    include/Particles.h
    include/Particles3D.h
    include/Particles3Dcomm.h
    include/phdf5.h
    include/PSKException.h
    include/PSKhdf5adaptor.h
    include/PSKOutput.h
    include/Restart3D.h
    include/TimeTasks.h
    include/Timing.h
    include/TransArraySpace3D.h
    include/VCtopology3D.h
    include/VirtualTopology3D.h
    inputoutput/Collective.cpp
    inputoutput/ParallelIO.cpp
    inputoutput/phdf5.cpp
    inputoutput/Restart3D.cpp
    main/iPic3Dlib.cpp
    mathlib/EllipticF.cpp
    particles/Particles3D.cpp
    particles/Particles3Dcomm.cpp
    performances/Timing.cpp
    PSKOutput3D/PSKhdf5adaptor.cpp
    solvers/CG.cpp
    solvers/GMRES.cpp
    utility/asserts.cpp
    utility/debug.cpp
    iPic3D.cpp utility/mathUtil.cpp include/mathUtil.h)

add_executable(iPic3D ${SOURCE_FILES})

target_link_libraries(iPic3D ${HDF5_LIB_PATH})
target_link_libraries(iPic3D ${MPICH2_LIB_PATH})