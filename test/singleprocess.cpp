#include "reduce.h"
#include <iostream>
#include <boost/random.hpp>
#include <vector>
#include <algorithm>
#include <cstdlib>

using namespace std;

void generateVector(int n, vector<double>& v)
{
    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<double> distr(1., 2.);

    v.resize(n);

    for (int i = 0; i < n; ++i)
        v[i] = distr(rng);
}

int main(int argc, char **argv)
{
    typedef double (*reducefunc_t)(int, const double*);
    reducefunc_t reducefunc = bitrep::reduce_3;

    if (argc > 1) {
        int arg = atoi(argv[1]);
        switch (arg) {
            case 1:
                reducefunc = bitrep::reduce_1;
                break;
            case 2:
                reducefunc = bitrep::reduce_2;
                break;
            case 3:
                reducefunc = bitrep::reduce_3;
                break;
            case 4:
                reducefunc = bitrep::reduce_4;
                break;
            default:
                cerr << "Invalid argument: " << arg << "\n";
                return 1;
        }
    }

    vector<double> input;

    for (int n = 1; n <= 1024; n *= 2) {
        generateVector(n, input);
        double result = reducefunc(n, input.data());

        for (int i = 0; i < 1024; ++i) {
            std::random_shuffle(input.begin(), input.end());
            double newresult = reducefunc(n, input.data());
            if (newresult != result)
                cerr << "Error at shuffle #" << i << endl;
        }
    }
}
