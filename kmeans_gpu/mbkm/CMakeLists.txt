
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project (Kmeans)

find_package(CUDA 8.0 REQUIRED)

option(CMAKE_VERBOSE_MAKEFILE ON)
option(DGTEST,  "DGTEST"  ON)

set(CUDA_NVCC_FLAGS -std=c++11)
set (CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/build/mbkm)

message("${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}")

set(KMEANS_SRC
  mbkm.cu
  ../include/cmdparser/cmdlineparser.cpp
  ../include/logger/logger.cpp
  ../include/mckm/mckm.cpp
  ../include/util/timer.cpp
  ../include/util/dataIo.cpp
  ../include/util/allocation.cpp
  ../include/util/arguments.cpp
)
set(KMEAN_HEADER
  ../include/cmdparser/cmdlineparser.h
  ../include/logger/logger.h
  ../include/mckm/mckm.h
  ../include/util/timer.h
  ../include/util/dataIo.h
  ../include/util/allocation.h
  ../include/util/arguments.h
)
set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)

cuda_add_executable(mbkm ${KMEANS_SRC} ${KMEAN_HEADER}
						OPTIONS ${GENCODE_SM70})