cmake_minimum_required(VERSION 3.18)
project(CUDA-Radar-SignalProcessing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# sm_60 (Pascal) and up, plus sm_120 (Blackwell, e.g. RTX 50-series) for the
# development machine this was built and validated on. Trim or extend this
# list to match your target GPUs -- a narrower list compiles faster.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90 120)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/cuda
)

add_executable(radar_cuda
  src/main.cu
  src/scenario.cpp
  src/cpu_reference/cpu_pipeline.cpp
  src/cuda/chirp_generation.cu
  src/cuda/matched_filter.cu
  src/cuda/range_doppler.cu
  src/cuda/coherent_integration.cu
  src/cuda/peak_detection.cu
)

target_link_libraries(radar_cuda PRIVATE CUDA::cudart CUDA::cufft)

set_target_properties(radar_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
  target_compile_options(radar_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2>)
endif()
