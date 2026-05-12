cmake_minimum_required(VERSION 4.2)
project(HNSW)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

set(SOURCES
        src/main.cpp
        src/vector_file_loader.cpp
        src/HNSW.cpp
        src/HNSW_benchmark.cpp
        src/progress_bar.cpp
        src/ground_truth_file_handler.cpp
)

add_executable(HNSW ${SOURCES})
target_include_directories(HNSW PUBLIC include)
set_target_properties(HNSW PROPERTIES COMPILE_FLAGS "-O3 -march=native -g")

if(OpenMP_CXX_FOUND)
    target_link_libraries(HNSW PUBLIC OpenMP::OpenMP_CXX)
endif()