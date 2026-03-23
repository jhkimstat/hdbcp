#ifndef CALIBRATION_COV_H
#define CALIBRATION_COV_H

#include <RcppArmadillo.h>

Rcpp::List calibrate_alpha_cov(const arma::vec& X_bar, const arma::mat& L,
                         int omega_k, double FPR_0, double C_cp,
                         int N_min, int N_batch, int N_max, double C_conv,
                         int n, int p, double trunc,
                         double b_I, double b_L, double b_R, double a0);

#endif
