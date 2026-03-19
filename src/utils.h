#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

void precompute_cumsums(const arma::mat& X, int n, int p, arma::mat& S, arma::mat& Q);

#endif
