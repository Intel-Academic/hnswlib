cmake_minimum_required (VERSION 2.6)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")



set(SOURCE_EXE main.cpp)           

set(SOURCE_LIB sift_1b.cpp)

add_library(sift_test STATIC ${SOURCE_LIB})


add_executable(main ${SOURCE_EXE})
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

add_executable(test_updates examples/updates_test.cpp)

target_link_libraries(main sift_test) 
