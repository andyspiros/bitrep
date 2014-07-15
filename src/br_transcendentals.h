// C++ interface: the functions are in the namespace bitrep
namespace bitrep
{
    double sin  (double x);
    double cos  (double x);
    double tan  (double x);
    double asin (double x);
    double acos (double x);
    double atan (double x);
    double sinh (double x);
    double cosh (double x);
    double tanh (double x);
    double asinh(double x);
    double acosh(double x);
    double atanh(double x);
    double log  (double x);
    double log1p(double x);
    double exp  (double x);
}

// C interface: we use the "br_" prefix
extern "C"
{
    double br_sin  (double x);
    double br_cos  (double x);
    double br_tan  (double x);
    double br_asin (double x);
    double br_acos (double x);
    double br_atan (double x);
    double br_sinh (double x);
    double br_cosh (double x);
    double br_tanh (double x);
    double br_asinh(double x);
    double br_acosh(double x);
    double br_atanh(double x);
    double br_log  (double x);
    double br_log1p(double x);
    double br_exp  (double x);
}

