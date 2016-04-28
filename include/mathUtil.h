//
// Created by vadim romansky on 28.04.16.
//

#ifndef IPIC3D_MATHUTIL_H
#define IPIC3D_MATHUTIL_H

double sqr(const double& a);
double cube(const double& a);
double solve4orderEquation(const double& a4, const double& a3, const double& a2, const double& a1, const double& a0, const double& startX);
double polynom4Value(const double& a4, const double& a3, const double& a2, const double& a1, const double& a0, const double& x);
double polynom4DerivativeValue(const double& a4, const double& a3, const double& a2, const double& a1, const double& x);

#endif //IPIC3D_MATHUTIL_H
