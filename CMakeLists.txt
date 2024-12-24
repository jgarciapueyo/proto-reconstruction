cmake_minimum_required(VERSION 3.22)

# Project name and a few useful settings. Other commands can pick up the results
project(
        proto-reconstruction
        VERSION 0.1
        DESCRIPTION "A prototype for reconstruction of 3D scenes based on Structure-from-Motion (SfM) and Visual Odometry (OD)"
        LANGUAGES CXX
)

# C++ Configuration
set(
        CMAKE_CXX_STANDARD 17
        CMAKE_CXX_STANDARD_REQUIRED ON
        CMAKE_CXX_EXTENSIONS OFF
)

add_executable(proto_reconstruction src/main.cpp)
