cmake_minimum_required (VERSION 3.2)

project (bitfusion)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)

if(!COMPILER_SUPPORTS_CXX20)
    message(FATAL_ERROR "Your compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Please use a different C++ compiler.")
endif()

find_package(CUDA)
find_package(MPI)
find_package(PkgConfig)

PKG_CHECK_MODULES(CPPUNIT REQUIRED cppunit)
PKG_CHECK_MODULES(NETCDF REQUIRED netcdf)
PKG_CHECK_MODULES(NETCDF_CXX4 REQUIRED netcdf-cxx4)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_NVCC_FLAGS} -use_fast_math -gencode arch=compute_50,code=sm_50 -gencode arch=compute_30,code=sm_30 -DOMPI_SKIP_MPICXX -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

set(ENGINE_DIR ../../src)
set(UTILS_DIR ../../src/utils)

include_directories(
    ${ENGINE_DIR}
    ${UTILS_DIR}
    ${CPPUNIT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_PATH}
    ${NETCDF_INCLUDE_DIR}
    ${NETCDF_CXX4_INCLUDE_DIR}
)

set(ENGINE_SOURCES
    ${ENGINE_DIR}/gpuTypes.cpp
    ${ENGINE_DIR}/kernels.cu
    ${ENGINE_DIR}/activation.cu
    ${ENGINE_DIR}/sorting.cu
    ${ENGINE_DIR}/sparse.cu
)

set(UTILS_SOURCES
    ${UTILS_DIR}/utils.cpp
)

set(TEST_SOURCES
    TestDune.cpp
)

cuda_add_executable(gputests
    ${ENGINE_SOURCES}
    ${TEST_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(gputests
    ${CPPUNIT_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${NETCDF_CXX4_LIBRARIES}
)
