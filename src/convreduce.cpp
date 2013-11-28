#include "convreduce.h"
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

#ifndef BITREP_MAXREP
# define BITREP_MAXREP 1000
#endif

/* Fortran interface for BLAS */
#ifdef BITREP_BLAS
extern "C"
{
    double dasum_(const int* n, const double* v, const int* inc);
}
#endif // BITREP_BLAS

/**
 * Helper class holding static information about the floating point types
 */
template<typename T> struct BitTraits;

template<> struct BitTraits<float>
{
    static const int maxExp = 127;
    static const int epsExp = -23;
    static const int mantissaBits = 23;
    static const int expBits = 8;

    static const int32_t expMask = 0x7f800000;

    typedef int32_t IntType;
    typedef uint32_t UintType;

    // template<unsigned int k>
    // static inline void kernelSIMD(int64_t n, int64_t N, const float* v,
    //                                float T[k], float M[k])
    // { andyReduce_double<k>(n, N, v, T, M); }

#ifdef BITREP_MPI
    static MPI_Datatype mpitype() { return MPI_FLOAT; }
#endif
};

template<> struct BitTraits<double>
{
    static const int maxExp = 1023;
    static const int epsExp = -52;
    static const int mantissaBits = 52;
    static const int expBits = 11;

    static const int64_t expMask = 0x7ff0000000000000;

    typedef double ScalarT;
    typedef int64_t IntType;
    typedef uint64_t UintType;

    // template<unsigned int k>
    // static inline void kernelSIMD(int64_t n, int64_t N, const double* v,
    //                                double T[k], double M[k])
    // { andyReduce_double<k>(n, N, v, T, M); }

#ifdef BITREP_MPI
    static MPI_Datatype mpitype() { return MPI_DOUBLE; }
#endif


    // SIMD utilities
#ifndef __AVX__
    typedef __m128d PackT;
    static const int packSize = 2;

    static inline PackT set1(ScalarT x) { return _mm_set1_pd(x); }
    static inline PackT load(const ScalarT* addr) { return _mm_loadu_pd(addr); }
    static inline void store(ScalarT* addr, PackT x) { _mm_storeu_pd(addr, x); }
    static inline ScalarT getFirst(PackT x) {
        ScalarT y; _mm_store_sd(&y, x);
        return y;
    }

    static inline long int cmpgt(PackT x, PackT y) {
        return _mm_movemask_pd(_mm_cmpgt_pd(x, y));
    }

    static inline PackT add(PackT x, PackT y) { return _mm_add_pd(x, y); }
    static inline PackT sub(PackT x, PackT y) { return _mm_sub_pd(x, y); }
    static inline PackT mul(PackT x, PackT y) { return _mm_mul_pd(x, y); }
    static inline PackT max(PackT x, PackT y) { return _mm_max_pd(x, y); }

#else // __AVX

    typedef __m256d PackT;
    static const int packSize = 4;

    static inline PackT set1(ScalarT x) { return _mm256_set1_pd(x); }
    static inline PackT load(const ScalarT* addr) { return _mm256_loadu_pd(addr); }
    static inline void store(ScalarT* addr, PackT x) { _mm256_storeu_pd(addr, x); }
    static inline ScalarT getFirst(PackT x) {
        ScalarT y;
        _mm_store_sd(&y, _mm256_castpd256_pd128(x));
        return y;
    }

    static inline long int cmpgt(PackT x, PackT y) {
        return _mm256_movemask_pd(_mm256_cmp_pd(x, y, _CMP_GT_OQ));
    }

    static inline PackT add(PackT x, PackT y) { return _mm256_add_pd(x, y); }
    static inline PackT sub(PackT x, PackT y) { return _mm256_sub_pd(x, y); }
    static inline PackT mul(PackT x, PackT y) { return _mm256_mul_pd(x, y); }
    static inline PackT max(PackT x, PackT y) { return _mm256_max_pd(x, y); }

#endif

    static inline ScalarT reducePack(PackT x) {
        ScalarT y[packSize], t = ScalarT(0.);
        store(y, x);

        for (int i = 0; i < packSize; ++i)
            t += y[i];

        return t;
    }

    static inline ScalarT maxPack(PackT x) {
        ScalarT y[packSize], t = ScalarT(0.);
        store(y, x);

        for (int i = 0; i < packSize; ++i)
            t = std::max(t, y[i]);

        return t;
    }
};


namespace bitrep
{

double convreduce(int n, const double* v)
{
#ifdef BITREP_BLAS
    static const int ONE = 1;
    return dasum_(&n, v, &ONE);
#else // BITREP_BLAS

    typedef BitTraits<double> Traits;

    Traits::PackT r_vec, T_vec;
    T_vec = Traits::set1(0.);

    // Allow pointer arithmetics
    const double* const lastP = v + (n / Traits::packSize * Traits::packSize);
    const int prefetch = 2048 / sizeof(double);

    // Main loop (TODO: handle odd n)
    while (v != lastP) {
        _mm_prefetch(v+prefetch, _MM_HINT_T1);
        r_vec = Traits::load(v);
        v += Traits::packSize;
        T_vec = Traits::add(T_vec, r_vec);
    }

    // Sum partial results
    return Traits::reducePack(T_vec);

#endif // BITREP_BLAS
}

#ifdef BITREP_MPI
double convreduce(int n, int N, const double* v, MPI_Comm comm)
{
    double localres = convreduce(n, v);
    double globalres;
    MPI_Reduce(&localres, &globalres, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    return globalres;
}

double convreduce_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    double localres, globalres;
    double e;
    int repeat;

    // Calculate repeat for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    localres = convreduce(n, v);
    tComp = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tComp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Local computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        localres = convreduce(n, v);
    tComp = (MPI_Wtime() - e) / repeat;

    // Calculate repeat for communication
    MPI_Barrier(comm);
    e = MPI_Wtime();
    MPI_Reduce(&localres, &globalres, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    tComm = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tComp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Time for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r) {
        MPI_Reduce(&localres, &globalres, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    tComm = (MPI_Wtime() - e) / repeat;

    return globalres;
}
#endif // BITREP_MPI

} // End namespace bitrep
