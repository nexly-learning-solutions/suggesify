cmake_minimum_required(VERSION 3.20)

project(sugesstify VERSION 1.0 LANGUAGES CXX)

# Enable C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPUNIT REQUIRED cppunit)
pkg_check_modules(NETCDF REQUIRED netcdf)
pkg_check_modules(NETCDF_CXX4 REQUIRED netcdf-cxx4)

# Define directories
set(ENGINE_DIR ../../src)
set(UTILS_DIR ../../src/utils)

# Define sources
set(UTILS_SOURCES
    ${UTILS_DIR}/netCDFhelper.cpp
    ${UTILS_DIR}/utils.cpp
)

set(TEST_SOURCES
    main.cpp
)

# Create the executable
add_executable(unittests
    ${TEST_SOURCES}
    ${UTILS_SOURCES}
)

# Specify include directories
target_include_directories(unittests PRIVATE
    ${ENGINE_DIR}
    ${UTILS_DIR}
    ${CPPUNIT_INCLUDE_DIRS}
    ${NETCDF_INCLUDE_DIRS}
    ${NETCDF_CXX4_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(unittests
    PRIVATE
    ${CPPUNIT_LIBRARIES}
    ${NETCDF_LIBRARIES}
    ${NETCDF_CXX4_LIBRARIES}
)