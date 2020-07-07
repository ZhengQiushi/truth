cmake_minimum_required(VERSION 3.15)
project(myCNN)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++11)


# Add OpenCV headers location to your include paths
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add Tensorflow headers location to your include paths
include_directories(./tensorflow/include)
include_directories(./tensorflow/include/src)

link_directories(./tensorflow)

# Declare the executable target built from your sources
add_executable(myCNN main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(myCNN ${OpenCV_LIBS})
# Link your application with Tensorflow libraries
target_link_libraries(myCNN /home/truth/ClionProjects/RM/bin/tensorflow/libtensorflow_cc.so /home/truth/ClionProjects/RM/bin/tensorflow/libtensorflow_framework.so.2)

