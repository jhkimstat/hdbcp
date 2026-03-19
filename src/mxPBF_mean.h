#ifndef MXPBF_MEAN_H
#define MXPBF_MEAN_H

#include <RcppArmadillo.h>

arma::vec compute_mxPBF_mean(const arma::mat& S, const arma::mat& Q,
                                  int s, int e, double alpha, int n, int p,
                                  double trunc);

#endif
