cmake_minimum_required(VERSION 3.22)

# Project name and a few useful settings. Other commands can pick up the results
project(
        proto-reconstruction
        VERSION 0.1
        DESCRIPTION "A prototype for reconstruction of 3D scenes based on Structure-from-Motion (SfM) and Visual Odometry (OD)"
        LANGUAGES CXX
)

# EXTERNAL PACKAGES
# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Sophus
include_directories(ext/Sophus)
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Pangolin
add_subdirectory(ext/Pangolin)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# C++ Configuration
set(
        CMAKE_CXX_STANDARD 17
        CMAKE_CXX_STANDARD_REQUIRED ON
        CMAKE_CXX_EXTENSIONS OFF
)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(proto_reconstruction
        src/utils/imagestream.cpp
        src/vo/frame.cpp
        src/vo/map.cpp
        src/vo/mappoint.cpp
        src/main.cpp
)
target_link_libraries(proto_reconstruction Eigen3::Eigen ${Sophus_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
