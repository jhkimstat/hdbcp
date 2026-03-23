#ifndef MXPBF_COV_H
#define MXPBF_COV_H

#include <RcppArmadillo.h>

arma::vec compute_mxPBF_cov(const arma::mat& X, int s, int e, double alpha, double trunc,
                                  double b_I, double b_L, double b_R, double a0);

#endif
