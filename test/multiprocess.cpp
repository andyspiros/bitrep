#ifndef BITREP_MPI
# error "This program requires the flag BITREP_MPI"
#endif

#include "mpi.h"
#include "reduce.h"
#include "convreduce.h"
#include <cstdio>
#include <vector>
#include <boost/random.hpp>

using namespace std;

vector<double> input;

template<int k>
inline double doReduce(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
{
    if (k > 0)
        return bitrep::singlesweep_timing<k>(n, N, input.data(), comm, tComp, tComm);
    else if (k < 0)
        return bitrep::doublesweep_timing<-k>(n, N, input.data(), comm, tComp, tComm);
    else
        return bitrep::convreduce_timing(n, N, input.data(), comm, tComp, tComm);
}

void generateVector(int n)
{
    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<double> distr(1., 2.);

    input.resize(n);

    for (int i = 0; i < n; ++i)
        input[i] = distr(rng);
}

template<int k>
void test(int n, int N, MPI_Comm comm, bool verbose, int repeat)
{
    double tComp, tComm, TComp = 0., TComm = 0.;
    double sComp = 0., sComm = 0.;
    doReduce<k>(n, N, input.data(), comm, tComp, tComm);

    for (int r = 0; r < repeat; ++r) {
        doReduce<k>(n, N, input.data(), comm, tComp, tComm);
        TComp += tComp;
        TComm += tComm;
        sComp += tComp*tComp;
        sComm += tComm*tComm;
    }

    TComp /= repeat;
    TComm /= repeat;
    sComp /= repeat;
    sComm /= repeat;

    sComp = sqrt(sComp - TComp*TComp);
    sComm = sqrt(sComm - TComm*TComm);

    if (verbose)
        printf("%8d | %9.5f (%09.5f) | %9.5f (%09.5f)\n", n, TComp*1000., sComp*1000., TComm*1000., sComm*1000.);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nproc, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    const bool root = pid == 0;

    int nproc2 = 1;
    for(int p = 1; 2*p <= nproc; p *= 2)
        nproc2 *= 2;

    const int nmax = 1 << 25;
    generateVector(nmax);

    // Choose test
    int testfuncid = 1;
    if (argc > 1)
        testfuncid = atoi(argv[1]);

    typedef void (*testfunc_t)(int, int, MPI_Comm, bool, int);
    testfunc_t testfuncs[] = {
        test<-4>,
        test<-3>,
        test<-2>,
        test<-1>,
        test< 0>,
        test< 1>,
        test< 2>,
        test< 3>,
        test< 4>
    };
    testfunc_t testfunc = testfuncs[testfuncid+4];
    const int k = testfuncid;
    if (root) {
        cout << nproc << " participating processes\n";

        if (k > 0)
            cout << "Testing single-sweep with " << k << " levels\n" << endl;
        else if (k == 0)
            cout << "Testing conventional sum\n" << endl;
        else if (k < 0)
            cout << "Testing double-sweep with " << -k << " levels\n" << endl;

        cout << "Local n  |    Computation time   |   Communication time\n";
        cout << "---------+-----------------------+----------------------\n";
    }

    for (int n = 1; n <= nmax; n *= 2) {
        int N = n * nproc;
        testfunc(n, N, MPI_COMM_WORLD, root, 10);
    }

    if (root)
    {
        cout << "--------------------------------------------------------\n\n\n";
    }

    MPI_Finalize();
}
