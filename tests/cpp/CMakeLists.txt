cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 11)

project(forestry_tests)

# Once we have Catch2 3.x in system repo, we can stop using git submodule and let cmake find it in system locations
#find_package(Catch2 3 REQUIRED)
add_subdirectory(include/Catch2)

add_executable(tests test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)