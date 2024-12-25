# Part 0: Set Up OpenCV
## Prerequisites
- **C++ compiler:** Ensure you have a C++ compiler like GCC or Clang installed.
- **CMake:** Install CMake (version 3.20 or higher recommended).

## Step 1: Install OpenCV
- **OpenCV:** Install OpenCV. You can either:
  - Use a precompiled OpenCV package (e.g., via a package manager like apt on Linux).
  - Build OpenCV from source.

We will use a precompiled OpenCV package:
```bash
sudo apt update
sudo apt install libopencv-dev
```

## Step 2: Configure CMake
Create a `CMakeLists.txt` file in the project root:
```cmake
cmake_minimum_required(VERSION 3.22)

# Project name and a few useful settings. Other commands can pick up the results
project(
        proto-reconstruction
        VERSION 0.1
        DESCRIPTION "A prototype for reconstruction of 3D scenes based on Structure-from-Motion (SfM) and Visual Odometry (OD)"
        LANGUAGES CXX
)

# EXTERNAL PACKAGES
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# C++ Configuration
set(
        CMAKE_CXX_STANDARD 17
        CMAKE_CXX_STANDARD_REQUIRED ON
        CMAKE_CXX_EXTENSIONS OFF
)

add_executable(proto_reconstruction src/main.cpp)
target_link_libraries(proto_reconstruction ${OpenCV_LIBS})
```

## Step 3: Write the program code
Create the `main.cpp` file inside the `src` directory to see which OpenCV version we have:
```c++
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  std::cout << "OpenCV version : " << CV_VERSION << std::endl;
  std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
  std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
  std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
  return 0;
}
```

## Step 4: Build and run the project
1. Create the build directory:
```bash
mkdir build && cd build
```
2. Configure the project using CMake:
```bash
cmake ..
```
3. Build the project:
```bash
make
```
4. Run the project:
```bash
./main
```