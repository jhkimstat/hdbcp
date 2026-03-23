#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>
#include "mxPBF_cov.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec compute_mxPBF_cov(const arma::mat& X, int s, int e, double alpha, double trunc,
                                  double b_I, double b_L, double b_R, double a0) {
  int n = X.n_rows;
  int p = X.n_cols;
  int n_I = e - s;

  // Truncation
  int trim = std::max(1, static_cast<int>(std::floor(n_I * trunc)));
  int l_start = s + trim;
  int l_end = e - trim;
  if (l_start > l_end) {
    return arma::vec({ -datum::inf, -1.0 });
  }

  double gamma = std::pow(std::max(n, p), -alpha);
  double const_term1 = 0.5 * std::log(gamma / (1.0 + gamma));

  // Pre-compute V_I
  arma::mat X_I = X.rows(s, e - 1);
  arma::mat V_I = X_I.t() * X_I;

  // Initialize V_L
  arma::mat X_L = X.rows(s, l_start - 1);
  arma::mat V_L = X_L.t() * X_L;

  double max_log_PBF = -arma::datum::inf;
  int best_l = -1;

  for (int l = l_start; l <= l_end; ++l) {

    int n_L = l - s;
    int n_R = e - l;

    arma::mat V_R = V_I - V_L;

    double const_term2 = std::lgamma(n_L / 2.0 + a0) + std::lgamma(n_R / 2.0 + a0) - std::lgamma(n_I / 2.0 + a0);
    double const_term3 = a0 * std::log((b_L * b_R) / (b_I * std::tgamma(a0)));
    double base_const = const_term1 + const_term2 + const_term3;

    double n_L_half_a0 = n_L / 2.0 + a0;
    double n_R_half_a0 = n_R / 2.0 + a0;
    double n_I_half_a0  = n_I / 2.0 + a0;

    for (int j = 1; j < p; ++j) {
      for (int i = 0; i < j; ++i) {
        double rss_L = V_L(i, i) - (V_L(i, j) * V_L(i, j)) / V_L(j, j);
        double rss_R = V_R(i, i) - (V_R(i, j) * V_R(i, j)) / V_R(j, j);
        double rss_I = V_I(i, i) - (V_I(i, j) * V_I(i, j)) / V_I(j, j);

        double log_PBF = base_const
                        - n_L_half_a0 * std::log(b_L + 0.5 * rss_L)
                        - n_R_half_a0 * std::log(b_R + 0.5 * rss_R)
                        + n_I_half_a0  * std::log(b_I  + 0.5 * rss_I);

          if (log_PBF > max_log_PBF) {
            max_log_PBF = log_PBF;
            best_l = l;
          }
      }
    }

    // Update V_L
    if (l < l_end) {
      arma::rowvec x_curr = X.row(l);
      V_L += x_curr.t() * x_curr;
    }
  }

  return arma::vec({ max_log_PBF, static_cast<double>(best_l) });
}
