cmake_minimum_required(VERSION 3.22)

project(SDDMM_GPU)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CUDA_ARCHITECTURES 86)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

file(GLOB INCLUDE_FILES "${SRC_DIR}/*.hpp" "${SRC_DIR}/*.h" "${SRC_DIR}/*.cuh")
file(GLOB SRC_FILES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c" "${SRC_DIR}/*.cu")

message(STATUS "Include files: ${INCLUDE_FILES}")
message(STATUS "Src files: ${SRC_FILES}")

add_executable(SDDMM_GPU ${SRC_FILES} ${INCLUDE_FILES})

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

target_include_directories(SDDMM_GPU PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(SDDMM_GPU PRIVATE ${CUDA_CUDART_LIBRARY})
target_link_libraries(SDDMM_GPU PRIVATE OpenMP::OpenMP_CXX)
