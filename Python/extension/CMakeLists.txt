cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ftemplate-depth=1024")

project(forestry)


#### PyBind11

add_subdirectory(include/pybind11)


#### Armadillo

option(BUILD_SHARED_LIBS "build shared library" OFF)
option(BUILD_SMOKE_TEST "build smoke test" OFF)
add_subdirectory(include/armadillo)



add_library(extension
    MODULE
        main.cpp
        api.cpp
        src/forestry.cpp
        src/forestryTree.cpp
        src/dataFrame.cpp
        src/treeSplitting.cpp
        src/RFNode.cpp
        src/sampling.cpp
        src/utils.cpp
)

target_link_libraries(extension PRIVATE pybind11::module pybind11::lto pybind11::windows_extras PUBLIC armadillo)

pybind11_extension(extension)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(extension)
endif()

set_target_properties(extension PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")

target_include_directories(extension PRIVATE src)
target_include_directories(extension PRIVATE src/rapidjson/include)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(extension PRIVATE VERSION_INFO=${VERSION_INFO})