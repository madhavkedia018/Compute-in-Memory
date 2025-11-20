cmake_minimum_required(VERSION 3.15)
project(vgg_libtorch_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/path/to/libtorch") # <-- change this to your libtorch path

find_package(Torch REQUIRED)

add_executable(vgg16_cnn vgg16_cnn.cpp)
target_link_libraries(vgg16_cnn "${TORCH_LIBRARIES}")
set_property(TARGET vgg16_cnn PROPERTY CXX_STANDARD 14)

add_executable(vgg16_bnn vgg16_bnn.cpp)
target_link_libraries(vgg16_bnn "${TORCH_LIBRARIES}")
set_property(TARGET vgg16_bnn PROPERTY CXX_STANDARD 14)

# for linux: link with pthread
if(UNIX)
  target_link_libraries(vgg16_cnn pthread)
  target_link_libraries(vgg16_bnn pthread)
endif()
