# 设置项目名称
project(Gemini)
cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 17)

set(DML_DIR ~/DML/install/lib/cmake/DML)
find_package(DML REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
add_subdirectory("Gimulator")

add_definitions(-DSTATISTICS)

set(CMAKE_CXX_FLAGS " -w -fsanitize=address -fno-sanitize=leak  -std=c++17 -g -fopenmp -march=native")
set(TARGETS bc bfs cc pagerank sssp )
# set(GIM_TARGETS gim_bc gim_bfs gim_cc gim_pagerank gim_sssp)
file(GLOB HEADERS "core/*.hpp")
set(GIM_FILES "Gimulator/src/DMA.cpp" "Gimulator/src/CXL_SHM.cpp" "Gimulator/src/communicate.cpp")

include_directories(${CMAKE_SOURCE_DIR}/core)
include_directories(${CMAKE_SOURCE_DIR}/Gimulator/include)

# 将每个目标的 C++ 文件转换为可执行文件
foreach(target ${TARGETS})
    set(SOURCE_FILE "gim_toolkits/${target}.cpp")
    # set(SOURCE_FILE "toolkits/${target}.cpp")
    add_executable(${target} ${SOURCE_FILE} ${HEADERS} ${GIM_FILES})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR})
    target_link_libraries(${target} MPI::MPI_CXX OpenMP::OpenMP_CXX numa dml dmlhl)
endforeach()

