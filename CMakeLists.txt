cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer native arch when building on a real GPU host; override with -DCMAKE_CUDA_ARCHITECTURES=75 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(vector_add 01_vector_add/vector_add.cu)
add_executable(matrix_add_2d 02_matrix_add_2d/matrix_add_2d.cu)

foreach(tgt vector_add matrix_add_2d)
  set_target_properties(${tgt} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
