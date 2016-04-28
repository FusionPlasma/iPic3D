#include <cmath>
#include "../include/mathUtil.h"

double sqr(const double& a){
    return a*a;
}

double cube(const double& a){
    return a*a*a;
}

//a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0
double solve4orderEquation(const double& a4, const double& a3, const double& a2, const double& a1, const double& a0, const double& startX) {
    double nextX = startX;
    double prevX = startX;

    int iterationCount = 0;
    while ((fabs(nextX - prevX) > fabs(prevX * 1.0E-20) || iterationCount == 0) && (iterationCount < 10000)) {
        prevX = nextX;
        nextX = prevX - polynom4Value(a4, a3, a2, a1, a0, prevX) / polynom4DerivativeValue(a4, a3, a2, a1, prevX);
        iterationCount++;
    }

    return nextX;
}

double polynom4Value(const double& a4, const double& a3, const double& a2, const double& a1, const double& a0, const double& x) {
    return (((a4 * x + a3) * x + a2) * x + a1) * x + a0;
}

double polynom4DerivativeValue(const double& a4, const double& a3, const double& a2, const double& a1, const double& x) {
    return ((4.0 * a4 * x + 3.0 * a3) * x + 2.0 * a2) * x + a1;
}
