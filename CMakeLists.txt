cmake_minimum_required(VERSION 3.10.0)
project(gtcrn VERSION 0.1.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_executable(gtcrn GTCRNImpl.cpp GTCRNImpl.h pocketfft_hdronly.h main.cpp)
target_link_libraries(gtcrn PUBLIC onnxruntime)
