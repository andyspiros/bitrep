#ifndef BITREP_MPI
# error "This program requires the flag BITREP_MPI"
#endif

#include "mpi.h"
#include "reduce.h"
#include <cstdio>
#include <vector>

using namespace std;

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

    cout << nproc2 << endl;

    MPI_Finalize();
}
