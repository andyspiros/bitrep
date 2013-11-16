#ifdef BITREP_MPI
# include "mpi.h"
#endif

/* Single-sweep functions */
namespace bitrep
{

    /* Local functions */
    double singlesweep_1(int n, const double* v);
    double singlesweep_2(int n, const double* v);
    double singlesweep_3(int n, const double* v);
    double singlesweep_4(int n, const double* v);

    template<int k> double singlesweep(int n, const double* v)
    {
        if (k == 1) return singlesweep_1(n, v);
        if (k == 2) return singlesweep_2(n, v);
        if (k == 3) return singlesweep_3(n, v);
        if (k == 4) return singlesweep_4(n, v);
    }

#ifdef BITREP_MPI

    /* Distributed functions */
    double singlesweep_1(int n, int N, const double* v, MPI_Comm comm);
    double singlesweep_2(int n, int N, const double* v, MPI_Comm comm);
    double singlesweep_3(int n, int N, const double* v, MPI_Comm comm);
    double singlesweep_4(int n, int N, const double* v, MPI_Comm comm);

    template<int k> double singlesweep(int n, int N, const double* v, MPI_Comm comm)
    {
        if (k == 1) return singlesweep_1(n, N, v, comm);
        if (k == 2) return singlesweep_2(n, N, v, comm);
        if (k == 3) return singlesweep_3(n, N, v, comm);
        if (k == 4) return singlesweep_4(n, N, v, comm);
    }


    /* Timing functions */
    double singlesweep_1_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double singlesweep_2_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double singlesweep_3_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double singlesweep_4_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);

    template<int k> double singlesweep_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
    {
        if (k == 1) return singlesweep_1_timing(n, N, v, comm, tComp, tComm);
        if (k == 2) return singlesweep_2_timing(n, N, v, comm, tComp, tComm);
        if (k == 3) return singlesweep_3_timing(n, N, v, comm, tComp, tComm);
        if (k == 4) return singlesweep_4_timing(n, N, v, comm, tComp, tComm);
    }
#endif

} // End namespace bitrep



 /* Double-sweep functions */
namespace bitrep
{

    /* Local functions */
    double doublesweep_1(int n, const double* v);
    double doublesweep_2(int n, const double* v);
    double doublesweep_3(int n, const double* v);
    double doublesweep_4(int n, const double* v);

    template<int k> double doublesweep(int n, const double* v)
    {
        if (k == 1) return doublesweep_1(n, v);
        if (k == 2) return doublesweep_2(n, v);
        if (k == 3) return doublesweep_3(n, v);
        if (k == 4) return doublesweep_4(n, v);
    }

#ifdef BITREP_MPI

    /* Distributed functions */
    double doublesweep_1(int n, int N, const double* v, MPI_Comm comm);
    double doublesweep_2(int n, int N, const double* v, MPI_Comm comm);
    double doublesweep_3(int n, int N, const double* v, MPI_Comm comm);
    double doublesweep_4(int n, int N, const double* v, MPI_Comm comm);

    template<int k> double doublesweep(int n, int N, const double* v, MPI_Comm comm)
    {
        if (k == 1) return doublesweep_1(n, N, v, comm);
        if (k == 2) return doublesweep_2(n, N, v, comm);
        if (k == 3) return doublesweep_3(n, N, v, comm);
        if (k == 4) return doublesweep_4(n, N, v, comm);
    }


    /* Timing functions */
    double doublesweep_1_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double doublesweep_2_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double doublesweep_3_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);
    double doublesweep_4_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm);

    template<int k> double doublesweep_timing(int n, int N, const double* v, MPI_Comm comm, double& tComp, double& tComm)
    {
        if (k == 1) return doublesweep_1_timing(n, N, v, comm, tComp, tComm);
        if (k == 2) return doublesweep_2_timing(n, N, v, comm, tComp, tComm);
        if (k == 3) return doublesweep_3_timing(n, N, v, comm, tComp, tComm);
        if (k == 4) return doublesweep_4_timing(n, N, v, comm, tComp, tComm);
    }
#endif

} // End namespace bitrep
