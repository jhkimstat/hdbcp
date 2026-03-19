#ifndef CALIBRATION_MEAN_H
#define CALIBRATION_MEAN_H

#include <RcppArmadillo.h>

Rcpp::List calibrate_alpha_mean(const arma::vec& bar_X, const arma::mat& L,
                          int omega_k, double FPR_0, double C_cp,
                          int N_min, int N_batch, int N_max, double C_conv,
                          int n, int p, double trunc);

#endif
