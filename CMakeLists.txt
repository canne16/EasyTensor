cmake_minimum_required(VERSION 3.14)

project(EasyTensor VERSION 1.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

configure_file(Config.h.in Config.h)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

set(SOURCES
${SRC_DIR}/main.cc
${SRC_DIR}/etc.cc
${SRC_DIR}/operations.cc
${SRC_DIR}/vizualizer.cc
)

add_executable(EasyTensor ${SOURCES})

target_include_directories(EasyTensor PUBLIC
                          ${INCLUDE_DIR}
                          "${PROJECT_BINARY_DIR}"
                          )

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0  # Specific version 1.16.0
)
FetchContent_MakeAvailable(googletest)

add_executable(runTests 
                        tests/test_main.cpp
              )

target_compile_definitions(runTests PRIVATE TESTING)


target_include_directories(runTests PUBLIC
                          ${INCLUDE_DIR}
                          "${PROJECT_BINARY_DIR}"
                          )

target_link_libraries(runTests gtest gtest_main pthread)


enable_testing()
add_test(NAME BasicTest COMMAND runTests)