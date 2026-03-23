#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include "mxPBF_mean.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec compute_mxPBF_mean(const arma::mat& S, const arma::mat& Q,
                             int s, int e, double alpha, int n, int p,
                             double trunc) {
  int n_I = e - s;
  double gamma = std::pow(std::max(n, p), -alpha);
  double const_term = 0.5 * std::log(gamma / (1.0 + gamma));

  double max_log_PBF = -datum::inf;
  int best_l = -1;

  // Truncation of searching interval
  int trim = std::max(1, static_cast<int>(std::floor(n_I * trunc)));
  int l_start = s + trim;
  int l_end = e - trim;
  if (l_start > l_end) {
    return arma::vec({ -datum::inf, -1.0 });
  }


  for (int l = l_start; l <= l_end; ++l) {
    int n_L = l - s;
    int n_R = e - l;
    double max_j_log_PBF = -datum::inf;

    for (int j = 0; j < p; ++j) {
      double sum_sq_I = Q(e, j) - Q(s, j);
      double sum_I_L  = S(l, j) - S(s, j);
      double sum_I_R  = S(e, j) - S(l, j);

      double num = sum_sq_I - (sum_I_L + sum_I_R) * (sum_I_L + sum_I_R) / n_I;
      double den = sum_sq_I - (sum_I_L * sum_I_L) / n_L - (sum_I_R * sum_I_R) / n_R;

      double log_PBF_j = const_term + (n_I / 2.0) * std::log(num / den);
      if (log_PBF_j > max_j_log_PBF) {
        max_j_log_PBF = log_PBF_j;
      }
    }

    if (max_j_log_PBF > max_log_PBF) {
      max_log_PBF = max_j_log_PBF;
      best_l = l;
    }
  }

  return arma::vec({ max_log_PBF, static_cast<double>(best_l) });
}
