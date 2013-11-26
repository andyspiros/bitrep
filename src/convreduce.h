#ifdef BITREP_MPI
# include "mpi.h"
#endif

/* Conventional (non-reproducible) sum functions */
namespace bitrep
{

    /* Local functions */
    double convreduce(int n, const double* v);

#ifdef BITREP_MPI

    /* Distributed functions */
    double convreduce(int n, int N, const double* v, MPI_Comm comm);

    /* Timing functions */
    double convreduce_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
#endif

} // End namespace bitrep

