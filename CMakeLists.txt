# 设置项目名称
project(Gemini)
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)
find_package(MPI REQUIRED)
add_subdirectory("Gimulator")

set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17 -g -fopenmp -march=native")
set(TARGETS bc bfs cc pagerank sssp )
# set(GIM_TARGETS gim_bc gim_bfs gim_cc gim_pagerank gim_sssp)
file(GLOB HEADERS "core/*.hpp")

include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/Gimulator/include)

# 将每个目标的 C++ 文件转换为可执行文件
foreach(target ${TARGETS})
    set(SOURCE_FILE "gim_toolkits/${target}.cpp")
    # set(SOURCE_FILE "toolkits/${target}.cpp")
    add_executable(${target} ${SOURCE_FILE} ${HEADERS})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${target} MPI::MPI_CXX numa)
endforeach()

