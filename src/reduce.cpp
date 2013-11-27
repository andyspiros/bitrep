#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

#ifdef BITREP_MPI
# include "mpi.h"
#endif

#ifndef BITREP_MAXREP
# define BITREP_MAXREP 1000
#endif

namespace bitrep
{

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



/********************
 * HELPER FUNCTIONS *
 ********************/


/**
 * Returns the parameter used by the singleSweep reduce
 */
template<typename T>
inline static void getLogAlpha(int n,
                 typename BitTraits<T>::IntType& logAlpha)
{
    typedef typename BitTraits<T>::IntType IntType;

    union
    {
        T f;
        IntType i;
    } alpha;

    alpha.f = static_cast<T>(n);

    // Extract exponent
    const IntType m = BitTraits<T>::mantissaBits, e = BitTraits<T>::expBits;
    logAlpha = (alpha.i >> m) % (IntType(1) << e) - BitTraits<T>::maxExp;

    // Check if mantissa is zero
    logAlpha += (alpha.i % (IntType(1) << m) != 0) ? 1 : 0;
}

template<typename T>
inline static void getOptExtractor(int n, T vmax, T& M, T& beta)
{
    typedef typename BitTraits<T>::IntType IntType;
    IntType e1, e2;
    getLogAlpha<T>(n, e1);

    union
    {
        T f;
        IntType i;
    } logM;
    logM.f = vmax;

    const IntType m = BitTraits<T>::mantissaBits, e = BitTraits<T>::expBits;
    e2 = (logM.i >> m) % (IntType(1) << e) - BitTraits<T>::maxExp;
    e2 += (logM.i % (IntType(1) << m) != 0) ? 1 : 0;

    // Build logM
    logM.i = (e1 + e2 + BitTraits<T>::maxExp) << m;
    M = logM.f;

    // Build beta
    logM.i = (e1 + BitTraits<T>::epsExp + BitTraits<T>::maxExp) << m;
    beta = logM.f;
}



/**
 * Returns the f-th M
 */
template<typename ScalarT>
inline static ScalarT getM(
        typename BitTraits<ScalarT>::IntType f,
        typename BitTraits<ScalarT>::IntType logAlpha)
{
    typedef typename BitTraits<ScalarT>::IntType IntType;

    union
    {
        ScalarT f;
        IntType i;
    } M;

    const IntType m = BitTraits<ScalarT>::mantissaBits,
                  e = BitTraits<ScalarT>::maxExp;
    M.i = (f*(IntType(2) - m + logAlpha) + e) + e << m;
    M.i |= IntType(1) << (m-1);

    return M.f;
}

/**
 * Returns the maximal supported value
 */
template<typename ScalarT>
inline static ScalarT getMaxv(int f, int logAlpha)
{
    union
    {
        ScalarT f;
        typename BitTraits<ScalarT>::IntType i;
    } maxv;

    maxv.i  =
              2*f
            + (f-1) * logAlpha
            + f * BitTraits<ScalarT>::epsExp
            + 2*BitTraits<ScalarT>::maxExp - 2;
    maxv.i <<= BitTraits<ScalarT>::mantissaBits;

    return maxv.f;
}

/**
 * Returns the smallest M supporting the value
 */
template<typename ScalarT>
inline static ScalarT getFirstM(
        ScalarT v,
        typename BitTraits<ScalarT>::IntType logAlpha,
        typename BitTraits<ScalarT>::IntType& h,
        ScalarT& maxv)
{
    typedef typename BitTraits<ScalarT>::IntType IntType;

    h = -1;
    ScalarT M, oldM;
    ScalarT oldmaxv;

    // Look for first suitable h
    do {
        ++h;

        oldmaxv = maxv;
        oldM = M;

        M = getM<ScalarT>(h, logAlpha);
        maxv = getMaxv<ScalarT>(h, logAlpha);
    } while(v <= maxv);

    maxv = oldmaxv;
    --h;

    return oldM;
}





/**********************
 * REDUCTION FUNCTION *
 **********************/



/**
 * Optimized version of the internal single sweep using SSE2 instructions
 * in double precision.
 *
 * \param n The size of local array
 * \param N The size of global array
 * \param v The array of size n
 * \param[out] T The array that will contain the sums of the k levels
 * \param[out] M The array that will contain the k extractors
 */
template<typename ScalarT, unsigned int k>
void singlesweep_kernel(int64_t n, int64_t N, const ScalarT* v, ScalarT T[k], ScalarT Ms[k])
{
    typedef BitTraits<ScalarT> Traits;
    typedef typename Traits::IntType IntType;
    const int k1 = k-1;

    IntType logAlpha;
    getLogAlpha<ScalarT>(N, logAlpha);

    // M: starting Ms
    // T: partial sums
    // S: additional contribution
    ScalarT S0;

    // Keep maxv, which is the exponent of the current maximal supported value
    ScalarT maxv;

    // Keep h, which is the index of the first M in M and T
    IntType h;

    // Vector registers
    typename Traits::PackT r_vec, cmp_vec, S_vec, S0_vec, Q_vec, maxv_vec,
                           T_vec0, T_vec1, T_vec2, T_vec3,
                           M_vec0, M_vec1, M_vec2, M_vec3;

    // Fill M, T, S; set maxv, h
    T_vec0 = Traits::set1(ScalarT(0.));
    M_vec0 = Traits::set1(getFirstM<ScalarT>(v[0], logAlpha, h, maxv));
    maxv_vec = Traits::set1(maxv);

    S0_vec = Traits::set1(getM<ScalarT>(h-1, logAlpha));
    S_vec = Traits::set1(ScalarT(0.));

    if (k > 1) {
        T_vec1 = Traits::set1(ScalarT(0.));
        M_vec1 = Traits::set1(getM<ScalarT>(h+1, logAlpha));
    }

    if (k > 2) {
        T_vec2 = Traits::set1(ScalarT(0.));
        M_vec2 = Traits::set1(getM<ScalarT>(h+2, logAlpha));
    }

    if (k > 3) {
        T_vec3 = Traits::set1(ScalarT(0.));
        M_vec3 = Traits::set1(getM<ScalarT>(h+3, logAlpha));
    }

    // Allow pointer arithmetics
    const ScalarT* const lastP = v + (n / Traits::packSize * Traits::packSize);
    const int prefetch = 2048 / sizeof(ScalarT);

    // Main loop (TODO: handle odd n)
    while (v != lastP) {
        _mm_prefetch(v+prefetch, _MM_HINT_T1);
        r_vec = Traits::load(v);
        v += Traits::packSize;

        // Check if current extractor supports both numbers
        long int compresult = Traits::cmpgt(r_vec, maxv_vec);
        if (__builtin_expect(compresult, 0)) {

            // Shift down sums and extractors
            if (k > 3) {
                T_vec3 = T_vec2;
                M_vec3 = M_vec2;
            }

            if (k > 2) {
                T_vec2 = T_vec1;
                M_vec2 = M_vec1;
            }

            // Subtract additional contribution
            if (k > 1) {
                T_vec1 = Traits::sub(T_vec0, S_vec);
                M_vec1 = M_vec0;
            }

            // Include additional contributions
            T_vec0 = S_vec;
            M_vec0 = S0_vec;

            // Add one more additional level
            --h;
            S0_vec = Traits::set1(getM<ScalarT>(h-1, logAlpha));
            S_vec = Traits::set1(ScalarT(0.));
            maxv_vec = Traits::set1(getMaxv<ScalarT>(h, logAlpha));
        }

        // Update additional constribution
        Q_vec = Traits::add(S0_vec, r_vec);
        Q_vec = Traits::sub(Q_vec, S0_vec);
        S_vec = Traits::add(S_vec, Q_vec);

        // Level 0
        if (k > 0) {
            Q_vec = Traits::add(M_vec0, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec0);
            T_vec0 = Traits::add(T_vec0, Q_vec);
            if (k > 1) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 1
        if (k > 1) {
            Q_vec = Traits::add(M_vec1, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec1);
            T_vec1 = Traits::add(T_vec1, Q_vec);
            if (k > 2) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 2
        if (k > 2) {
            Q_vec = Traits::add(M_vec2, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec2);
            T_vec2 = Traits::add(T_vec2, Q_vec);
            if (k > 3) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 3
        if (k > 3) {
            Q_vec = Traits::add(M_vec3, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec3);
            T_vec3 = Traits::add(T_vec3, Q_vec);
            //if (k > 4) r_vec = Traits::sub(r_vec, Q_vec);
        }
    }

    // Sum the elements in T_vec
    if (k > 0) {
        Ms[0] = Traits::getFirst(M_vec0);
        T[0] = Traits::reducePack(T_vec0) + Ms[0];
    }

    if (k > 1) {
        Ms[1] = Traits::getFirst(M_vec1);
        T[1] = Traits::reducePack(T_vec1) + Ms[1];
    }

    if (k > 2) {
        Ms[2] = Traits::getFirst(M_vec2);
        T[2] = Traits::reducePack(T_vec2) + Ms[2];
    }

    if (k > 3) {
        Ms[3] = Traits::getFirst(M_vec3);
        T[3] = Traits::reducePack(T_vec3) + Ms[3];
    }
}


template<typename ScalarT, unsigned k>
ScalarT singlesweep_internal(int64_t n, const ScalarT* v)
{
    ScalarT T[k], M[k];
    singlesweep_kernel<ScalarT, k>(n, n, v, T, M);

    ScalarT t = ScalarT(0.);
    for (int f = k-1; f >= 0; --f) {
        t += T[f] - M[f];
    }

    return t;
}

double singlesweep_1(int n, const double *v)
{
    return singlesweep_internal<double, 1>(n, v);
}

double singlesweep_2(int n, const double *v)
{
    return singlesweep_internal<double, 2>(n, v);
}

double singlesweep_3(int n, const double *v)
{
    return singlesweep_internal<double, 3>(n, v);
}

double singlesweep_4(int n, const double *v)
{
    return singlesweep_internal<double, 4>(n, v);
}


/***********************
 * DOUBLE-SWEEP REDUCE *
 ***********************/

/**
 * Compute max value in array using SIMD instructions
 */
double maxReduce(int n, const double *v)
{
    typedef double ScalarT;
    typedef BitTraits<ScalarT> Traits;
    typedef BitTraits<ScalarT>::PackT PackT;

    // Vector registers
    PackT r_vec, Q_vec;

    /* Find vmax */
    Q_vec = Traits::set1(0.);

    // Allow pointer arithmetics
    const ScalarT* const lastP = v + (n / Traits::packSize * Traits::packSize);
    const int prefetch = 2048 / sizeof(ScalarT);

    while (v != lastP) {
        _mm_prefetch(v+prefetch, _MM_HINT_T1);
        r_vec = Traits::load(v);
        v += Traits::packSize;

        r_vec = Traits::mul(r_vec, r_vec);
        Q_vec = Traits::max(r_vec, Q_vec);
    }
    return std::sqrt(Traits::maxPack(Q_vec));
}

/**
 * Optimized version of the internal double sweep using SIMD instructions
 *
 * \param n The size of local array
 * \param N The size of global array
 * \param v The array of size n
 * \param[out] T The array that will contain the sums of the k levels
 * \param[out] M The array that will contain the k extractors
 */
template<typename ScalarT, unsigned int k>
void doublesweep_internal(int64_t n, int64_t N, const double *v, double T[k], double Ms[k], double vmax)
{
    typedef BitTraits<ScalarT> Traits;
    typedef typename Traits::IntType IntType;

    // Vector registers
    typename Traits::PackT r_vec, Q_vec,
                           T_vec0, T_vec1, T_vec2, T_vec3,
                           M_vec0, M_vec1, M_vec2, M_vec3;

    /* Initialize vectors */
    double beta, M_cur;
    getOptExtractor<ScalarT>(n, vmax, M_cur, beta);
    M_vec0 = Traits::set1(M_cur);
    T_vec0 = Traits::set1(0.);
    if (k > 1) {
        M_vec1 = Traits::set1(M_cur *= beta);
        T_vec1 = Traits::set1(0.);
    }
    if (k > 2) {
        M_vec2 = Traits::set1(M_cur *= beta);
        T_vec2 = Traits::set1(0.);
    }
    if (k > 3) {
        M_vec3 = Traits::set1(M_cur *= beta);
        T_vec3 = Traits::set1(0.);
    }

    // Allow pointer arithmetics
    const ScalarT* const lastP = v + (n / Traits::packSize * Traits::packSize);
    const int prefetch = 2048 / sizeof(ScalarT);

    /* Main loop (TODO: support odd n) */
    while (v != lastP) {
        _mm_prefetch(v+prefetch, _MM_HINT_T1);
        r_vec = Traits::load(v);
        v += Traits::packSize;

        // Level 0
        if (k > 0) {
            Q_vec = Traits::add(M_vec0, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec0);
            T_vec0 = Traits::add(T_vec0, Q_vec);
            if (k > 1) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 1
        if (k > 1) {
            Q_vec = Traits::add(M_vec1, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec1);
            T_vec1 = Traits::add(T_vec1, Q_vec);
            if (k > 2) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 2
        if (k > 2) {
            Q_vec = Traits::add(M_vec2, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec2);
            T_vec2 = Traits::add(T_vec2, Q_vec);
            if (k > 3) r_vec = Traits::sub(r_vec, Q_vec);
        }

        // Level 3
        if (k > 3) {
            Q_vec = Traits::add(M_vec3, r_vec);
            Q_vec = Traits::sub(Q_vec, M_vec3);
            T_vec3 = Traits::add(T_vec3, Q_vec);
            //if (k > 4) r_vec = Traits::sub(r_vec, Q_vec);
        }
    }

    // Sum the elements in T_vec
    if (k > 0) {
        Ms[0] = Traits::getFirst(M_vec0);
        T[0] = Traits::reducePack(T_vec0) + Ms[0];
    }

    if (k > 1) {
        Ms[1] = Traits::getFirst(M_vec1);
        T[1] = Traits::reducePack(T_vec1) + Ms[1];
    }

    if (k > 2) {
        Ms[2] = Traits::getFirst(M_vec2);
        T[2] = Traits::reducePack(T_vec2) + Ms[2];
    }

    if (k > 3) {
        Ms[3] = Traits::getFirst(M_vec3);
        T[3] = Traits::reducePack(T_vec3) + Ms[3];
    }
}


template<typename ScalarT, unsigned k>
ScalarT doublesweep(int64_t n, const ScalarT* v)
{
    double vmax = maxReduce(n, v);

    ScalarT T[k], M[k];
    doublesweep_internal<ScalarT, k>(n, n, v, T, M, vmax);

    ScalarT t = ScalarT(0.);
    for (int f = k-1; f >= 0; --f) {
        t += T[f] - M[f];
    }

    return t;
}


double doublesweep_1(int n, const double* v)
{
    return doublesweep<double, 1>(n, v);
}

double doublesweep_2(int n, const double* v)
{
    return doublesweep<double, 2>(n, v);
}

double doublesweep_3(int n, const double* v)
{
    return doublesweep<double, 3>(n, v);
}

double doublesweep_4(int n, const double* v)
{
    return doublesweep<double, 4>(n, v);
}



#ifdef BITREP_MPI

/**********************
 * MPI IMPLEMENTATION *
 **********************/

/*
 * Returns
 *   0 if both exponents are equal
 *   +1 if the exponent of a is larger
 *   -1 if the exponent of b is larger
 */
template<typename ScalarT>
int compareExp(ScalarT a, ScalarT b)
{
    union
    {
        ScalarT f;
        typename BitTraits<ScalarT>::IntType i;
    } aa, bb;
    aa.f = a;
    bb.f = b;

    aa.i >>= BitTraits<ScalarT>::mantissaBits;
    bb.i >>= BitTraits<ScalarT>::mantissaBits;

    if (aa.i > bb.i) return 1;
    if (aa.i < bb.i) return -1;
    return 0;
}

template<typename ScalarT>
void mergesumInternal(ScalarT* invec, ScalarT* inoutvec, int k)
{
    int offset = 0, o;

    do {
        o = compareExp(invec[std::max(0, offset)],
                       inoutvec[std::max(0, -offset)]);
        offset += o;
    } while (o != 0 && offset < k && -offset < k);

    union
    {
        ScalarT f;
        typename BitTraits<ScalarT>::IntType i;
    } M;
    const int m = BitTraits<ScalarT>::mantissaBits - 1;

    // invec has larger exponents
    if (offset >= 0) {
        for (int f = offset; f < k; ++f) {
            M.f = invec[f];
            ((M.i >>= m) |= 1) <<= m;

            invec[f] += inoutvec[f-offset] - M.f;
        }
        for (int f = 0; f < k; ++f)
            inoutvec[f] = invec[f];
        return;
    }

    // inoutvec has larger exponents
    for (int f = -offset; f < k; ++f) {
        M.f = invec[f];
        ((M.i >>= m) |= 1) <<= m;

        inoutvec[f+offset] += invec[f] - M.f;
    }
}

void mergesum(void *invec, void *inoutvec, int *len, MPI_Datatype *type)
{
    if (*type == MPI_FLOAT) {
        mergesumInternal(
            reinterpret_cast<float*>(invec),
            reinterpret_cast<float*>(inoutvec),
            *len
        );
        return;
    }

    if (*type == MPI_DOUBLE) {
        mergesumInternal(
            reinterpret_cast<double*>(invec),
            reinterpret_cast<double*>(inoutvec),
            *len
        );
        return;
    }
}



template<typename ScalarT, unsigned k>
ScalarT singlesweep_MPI(int64_t n, int64_t N, const ScalarT* v, MPI_Comm comm)
{
    ScalarT MM[k], T[k], Ts[k];

    singlesweep_kernel<ScalarT, k>(n, N, v, T, MM);

    // Register custom vector sum operator
    MPI_Op MPI_MERGESUM;
    MPI_Op_create(mergesum, 1, &MPI_MERGESUM);

    // Perform communication
    MPI_Reduce(T, Ts, k, BitTraits<ScalarT>::mpitype(), MPI_MERGESUM, 0, comm);

    // Free operator
    MPI_Op_free(&MPI_MERGESUM);

    // Final reduction (significant only to root)
    union
    {
        ScalarT f;
        typename BitTraits<ScalarT>::IntType i;
    } M;
    const int m = BitTraits<ScalarT>::mantissaBits - 1;

    ScalarT t = 0.f;
    for (int f = 0; f < k; ++f) {
        M.f = Ts[f];
        ((M.i >>= m) |= 1) <<= m;
        t += Ts[f] - M.f;
    }

    return t;
}

double singlesweep_1(int n, int N, const double* v, MPI_Comm comm)
{
    return singlesweep_MPI<double, 1>(n, N, v, comm);
}

double singlesweep_2(int n, int N, const double* v, MPI_Comm comm)
{
    return singlesweep_MPI<double, 2>(n, N, v, comm);
}

double singlesweep_3(int n, int N, const double* v, MPI_Comm comm)
{
    return singlesweep_MPI<double, 3>(n, N, v, comm);
}

double singlesweep_4(int n, int N, const double* v, MPI_Comm comm)
{
    return singlesweep_MPI<double, 4>(n, N, v, comm);
}

template<typename ScalarT, unsigned k>
ScalarT singlesweep_MPI_timing(int64_t n, int64_t N, const ScalarT* v,
        MPI_Comm comm, double& tComp, double& tComm)
{
    ScalarT MM[k], T[k], Ts[k];
    double e;
    int repeat;

    // Calculate repeat for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    singlesweep_kernel<ScalarT, k>(n, N, v, T, MM);
    tComp = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tComp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Time for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        singlesweep_kernel<ScalarT, k>(n, N, v, T, MM);
    tComp = (MPI_Wtime() - e) / repeat;

    // Register custom vector sum operator
    MPI_Op MPI_MERGESUM;
    MPI_Op_create(mergesum, 1, &MPI_MERGESUM);

    // Calculate repeat for communication
    MPI_Barrier(comm);
    e = MPI_Wtime();
    MPI_Reduce(T, Ts, k, BitTraits<ScalarT>::mpitype(), MPI_MERGESUM, 0, comm);
    tComm = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tComp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Time for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r) {
        MPI_Reduce(T, Ts, k, BitTraits<ScalarT>::mpitype(), MPI_MERGESUM, 0, comm);
    }
    tComm = (MPI_Wtime() - e) / repeat;

    // Free operator
    MPI_Op_free(&MPI_MERGESUM);

    // Final reduction (significant only to root)
    union
    {
        ScalarT f;
        typename BitTraits<ScalarT>::IntType i;
    } M;
    const int m = BitTraits<ScalarT>::mantissaBits - 1;

    ScalarT t = 0.f;
    for (int f = 0; f < k; ++f) {
        M.f = Ts[f];
        ((M.i >>= m) |= 1) <<= m;
        t += Ts[f] - M.f;
    }

    return t;
}

double singlesweep_1_timing(int n, int N, const double* v, MPI_Comm comm,
        double& tComp, double& tComm)
{
    return singlesweep_MPI_timing<double, 1>(n, N, v, comm, tComp, tComm);
}

double singlesweep_2_timing(int n, int N, const double* v, MPI_Comm comm,
        double& tComp, double& tComm)
{
    return singlesweep_MPI_timing<double, 2>(n, N, v, comm, tComp, tComm);
}

double singlesweep_3_timing(int n, int N, const double* v, MPI_Comm comm,
        double& tComp, double& tComm)
{
    return singlesweep_MPI_timing<double, 3>(n, N, v, comm, tComp, tComm);
}

double singlesweep_4_timing(int n, int N, const double* v, MPI_Comm comm,
        double& tComp, double& tComm)
{
    return singlesweep_MPI_timing<double, 4>(n, N, v, comm, tComp, tComm);
}



template<typename ScalarT, unsigned k>
ScalarT doublesweepMPI_timing(int64_t n, int64_t N, const ScalarT* v,
                             MPI_Comm comm, double& ecomp, double& ecomm)
{
    double e;
    double vmax, vmax_local;
    double T[k], M[k], Tglobal[k];
    double tcomp, tcomm;
    int repeat;
    ecomp = ecomm = 0.;

    // Local max
    MPI_Barrier(comm);
    e = MPI_Wtime();
    vmax_local = maxReduce(n, v);
    tcomp = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tcomp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);
    
    // Time for local max
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        vmax_local = maxReduce(n, v);
    ecomp += (MPI_Wtime() - e) / repeat;

    // Global max
    MPI_Barrier(comm);
    e = MPI_Wtime();
    MPI_Allreduce(&vmax_local, &vmax, 1, BitTraits<ScalarT>::mpitype(), MPI_SUM, comm);
    tcomm = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tcomp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);
    
    // Time for Global max
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        MPI_Allreduce(&vmax_local, &vmax, 1, BitTraits<ScalarT>::mpitype(), MPI_SUM, comm);
    ecomm += (MPI_Wtime() - e) / repeat;

    // Calculate repeat for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    doublesweep_internal<ScalarT, k>(n, N, v, T, M, vmax);
    tcomp = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tcomp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Time for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        doublesweep_internal<ScalarT, k>(n, N, v, T, M, vmax);
    ecomp += (MPI_Wtime() - e) / repeat;
    
    // Calculate repeat for communication
    MPI_Barrier(comm);
    e = MPI_Wtime();
    MPI_Reduce(T, Tglobal, k, BitTraits<ScalarT>::mpitype(), MPI_SUM, 0, comm);
    tcomm = MPI_Wtime() - e;
    repeat = std::max(5, std::min(BITREP_MAXREP, (int)std::ceil(0.1 / tcomp)));
    MPI_Bcast(&repeat, 1, MPI_INT, 0, comm);

    // Time for computation
    MPI_Barrier(comm);
    e = MPI_Wtime();
    for (int r = 0; r < repeat; ++r)
        MPI_Reduce(T, Tglobal, k, BitTraits<ScalarT>::mpitype(), MPI_SUM, 0, comm);
    ecomm += (MPI_Wtime() - e) / repeat;

    // Compute final result
    double res = 0.;
    for (int f = 0; f < k; ++f)
        res += Tglobal[f];

    return res;
}


double doublesweep_1_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    return doublesweepMPI_timing<double, 1>(n, N, v, comm, tComp, tComm);
}

double doublesweep_2_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    return doublesweepMPI_timing<double, 2>(n, N, v, comm, tComp, tComm);
}

double doublesweep_3_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    return doublesweepMPI_timing<double, 3>(n, N, v, comm, tComp, tComm);
}

double doublesweep_4_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    return doublesweepMPI_timing<double, 4>(n, N, v, comm, tComp, tComm);
}

#endif // BITREP_MPI

}

