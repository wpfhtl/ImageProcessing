cmake_minimum_required(VERSION 3.7)
project(HistEquNPP)

set(CMAKE_CXX_STANDARD 11)

include(/usr/share/cmake-3.10/Modules/FindCUDA.cmake)
#include_directories(./)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(CUDA REQUIRED COMPONENTS cufft)
find_package(CUDA REQUIRED)
message(STATUS "CUDA Dris : ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp histEqu.cu)
#cuda_add_executable(HistEquNPP ${SOURCE_FILES})
cuda_add_executable(HistEquNPP ${SOURCE_FILES})

link_directories(/usr/local/cuda/lib64/)
set(CUDA_DIRS /usr/local/cuda/lib64)

target_link_libraries(HistEquNPP ${OpenCV_LIBS})
#message(STATUS "CUDA Libs : ${CUDA_LIBRARIES}")
target_link_libraries(HistEquNPP ${CUDA_LIBRARIES})
target_link_libraries(HistEquNPP libnppc.so)
target_link_libraries(HistEquNPP libnppial.so libnppicc.so libnppicom.so libnppidei.so libnppif.so libnppig.so libnppim.so libnppist.so libnppisu.so libnppitc.so libnpps.so)
target_link_libraries(HistEquNPP libcufft.so)

