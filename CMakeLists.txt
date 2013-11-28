cmake_minimum_required (VERSION 2.8)
project (bitrep)
enable_language (CXX)

find_package (MPI REQUIRED)
if (MPI_CXX_LIBRARIES)
    set (mpilib ${MPI_CXX_LIBRARIES})
else (MPI_CXX_LIBRARIES)
    set (mpilib "")
endif (MPI_CXX_LIBRARIES)

add_definitions ("-DBITREP_MPI")
include_directories ("src")

set (REDUCELIB_SOURCES
    "./src/reduce.cpp"
    "./src/convreduce.cpp"
    )
add_library (bitrepreduce STATIC "src/reduce.cpp" "src/convreduce.cpp")
set_property (TARGET bitrepreduce PROPERTY COMPILE_FLAGS "-mavx")

add_executable (benchmark "./test/multiprocess.cpp")
target_link_libraries (benchmark bitrepreduce ${MPI_CXX_LIBRARIES})
