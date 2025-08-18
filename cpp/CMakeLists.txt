cmake_minimum_required(VERSION 3.16)
project(bigmeans LANGUAGES CXX)

# Language standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(Eigen3 REQUIRED) # Require Eigen3
find_package(OpenMP REQUIRED) # Require Eigen3

# Library
add_library(bigmeans bigmeans.cpp)
target_include_directories(bigmeans PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # for "bigmeans.hpp"
)
target_link_libraries(bigmeans PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(bigmeans PRIVATE -O3)

# Executable
add_executable(demo demo.cpp)
target_link_libraries(demo PRIVATE bigmeans)





