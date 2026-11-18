cmake_minimum_required(VERSION 3.20)
project(HNSW LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

set(HNSW_CORE_SOURCES
        src/hnsw/HNSW.cpp
        src/hnsw/progress_bar.cpp
)

set(HNSW_IO_SOURCES
        src/hnsw/io/dataset_file_loader.cpp
        src/hnsw/io/ground_truth_dataset_loader.cpp
        src/hnsw/io/vector_dataset_loader.cpp
)

set(HNSW_BENCHMARK_SOURCES
        src/hnsw/HNSW_benchmark.cpp
)

add_library(hnsw_lib STATIC
        ${HNSW_CORE_SOURCES}
        ${HNSW_IO_SOURCES}
        ${HNSW_BENCHMARK_SOURCES}
)

target_include_directories(hnsw_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hnsw_lib PUBLIC
        OpenMP::OpenMP_CXX
)

target_compile_options(hnsw_lib PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG>
        $<$<CONFIG:Debug>:-O0 -g>
)

add_executable(HNSW src/main.cpp)
target_link_libraries(HNSW PRIVATE hnsw_lib)