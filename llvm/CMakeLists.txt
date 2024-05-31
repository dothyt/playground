cmake_minimum_required(VERSION 3.14)
project(remote_tagging)

set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM CONFIG)
if(NOT "${LLVM_VERSION_MAJOR}" VERSION_EQUAL 15)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 15")
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

add_library(remote_tagging SHARED remote_tagging.cpp)
