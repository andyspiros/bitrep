cmake_minimum_required (VERSION 2.8)
project (bitrep)
enable_language (CXX)

find_package(MPI REQUIRED)

set (REDUCELIB_SOURCES
    "./src/reduce.cpp"
    "./src/convreduce.cpp"
    )

add_definitions("-DBITREP_MPI")
include_directories("src")

add_library (bitrepreduce STATIC ${REDUCELIB_SOURCES})

add_executable (benchmark "./test/multiprocess.cpp")
target_link_libraries (benchmark bitrepreduce ${MPI_CXX_LIBRARIES})
