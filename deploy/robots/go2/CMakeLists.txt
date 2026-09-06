cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
find_package(OpenCV REQUIRED)
add_executable(fetch_ctrl main_fetch.cpp)
target_link_libraries(fetch_ctrl ${OpenCV_LIBS})
target_include_directories(fetch_ctrl PRIVATE ${OpenCV_INCLUDE_DIRS})
