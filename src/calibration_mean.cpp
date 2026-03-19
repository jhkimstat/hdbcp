#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>
#include "mxPBF_mean.h"
#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List calibrate_alpha_mean(const arma::vec& X_bar, const arma::mat& L,
                          int omega_k, double FPR_0, double C_cp,
                          int N_min, int N_batch, int N_max, double C_conv,
                          int n, int p, double trunc) {

  int s = 0;
  arma::vec log_pbf_vals(N_max);

  arma::vec q_hist(static_cast<int>(1 + std::ceil(static_cast<double>(N_max - N_min) / N_batch)) );
  int q_count = 0;

  double q_target = 1.0 - FPR_0;
  arma::vec prob_vec = { q_target };

  arma::mat S_null(omega_k + 1, p, arma::fill::zeros);
  arma::mat Q_null(omega_k + 1, p, arma::fill::zeros);

  while (s < N_max) {
    int current_batch = (s == 0) ? N_min : std::min(N_batch, N_max - s);

    for (int i = 0; i < current_batch; ++i) {
      arma::mat Z = arma::randn<arma::mat>(omega_k, p);
      arma::mat X_null = Z * L;
      X_null.each_row() += X_bar.t();

      precompute_cumsums(X_null, omega_k, p, S_null, Q_null);

      arma::vec res = compute_mxPBF_mean(S_null, Q_null, 0, omega_k,
                                         0.0, n, p, trunc);

      log_pbf_vals(s + i) = res[0];
    }

    s += current_batch;


    double q_current = arma::as_scalar(arma::quantile(log_pbf_vals.head(s), prob_vec));
    q_hist(q_count++) = q_current;


    // early stopping
    if (q_count >= 3) {
      double q_s_2b = q_hist(q_count - 3);
      double q_s_1b = q_hist(q_count - 2);
      double q_s    = q_hist(q_count - 1);

      double diff1 = std::abs(q_s_2b - q_s_1b);
      double diff2 = std::abs(q_s_1b - q_s);

      if (std::max(diff1, diff2) < C_conv) {
        break;
      }
    }
  }

  // final alpha value
  double q_final = q_hist(q_count - 1);

  double loggam = C_cp + 0.5 * std::log(0.5) - q_final;
  double alpha = 0.0;

  if (loggam < 0) {
    double num = 2.0 * loggam - std::log(1.0 - std::exp(2.0 * loggam));
    double den = std::log(std::max(std::max(n, p), 2));
    alpha = - num / den;
  }

  return List::create(Named("alpha_hat") = alpha,
                      Named("total_simulation") = s);
}
