cmake_minimum_required (VERSION 2.8)
project (bitrep)
enable_language (CXX)

# Find MPI
find_package (MPI REQUIRED)
if (MPI_CXX_LIBRARIES)
    set (mpilib ${MPI_CXX_LIBRARIES})
    add_definitions ("-DBITREP_MPI")
    include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})
else (MPI_CXX_LIBRARIES)
    set (mpilib "")
endif (MPI_CXX_LIBRARIES)

# Configure AVX
set (BITREP_USE_AVX "OFF" CACHE BOOL "Whether to use AVX instructions")
if (BITREP_USE_AVX)
    set(AVX_FLAG "-mavx")
else (BITREP_USE_AVX)
    set(AVX_FLAG "-mno-avx")
endif (BITREP_USE_AVX)
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${AVX_FLAG}" )

# Configure FMA
set (BITREP_USE_FMA "OFF" CACHE BOOL "Whether to use FMA instructions")
if (BITREP_USE_FMA)
    set(FMA_FLAG "-mfma")
    add_definitions ("-DBITREP_FMA")
else (BITREP_USE_FMA)
    set(FMA_FLAG "-mno-fma")
endif (BITREP_USE_FMA)
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${FMA_FLAG}" )

include_directories ("src")

# Reduce
set (REDUCELIB_SOURCES
    "./src/reduce.cpp"
    "./src/convreduce.cpp"
    )
add_library (bitrepreduce STATIC ${REDUCELIB_SOURCES})

# Functions
set (TRASCENDLIB_SOURCES
    "./src/br_transcendentals.cpp"
    )
add_library (bitreptrascend STATIC ${TRASCENDLIB_SOURCES})

#add_executable (benchmark "./test/multiprocess.cpp")
#target_link_libraries (benchmark bitrepreduce ${mpilib})
