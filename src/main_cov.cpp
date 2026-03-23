#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "intervals.h"
#include "mxPBF_cov.h"
#include "calibration_cov.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Structure for detecting change points
struct Candidate {
  int id;           // interval index
  double pbf_0;     // mxPBF with alpha = 0
  int eta;          // estimated location
  int s;            // start point
  int e;            // end point

  // sorting criterion
  bool operator<(const Candidate& other) const {
    return pbf_0 > other.pbf_0;
  }
};

// Structure for output
struct CP_Result {
  int eta;          // estimated location
  double alpha;     // calibrated alpha
  double pbf;       // mxPBF value
  int s;            // start point
  int e;            // end point

  // sorting criterion
  bool operator<(const CP_Result& other) const {
    return eta < other.eta;
  }
};

// [[Rcpp::export]]
List main_cov(const arma::mat& X, double beta, double m_min,
              double FPR_0, double C_cp, int N_min, int N_batch,
              int N_max, double C_conv, double trunc,
              double b_I, double b_L, double b_R, double a0,
              int n_parallel = 1) {

  int n = X.n_rows;
  int p = X.n_cols;

  // Sample mean, variance, and its Cholesky decomposition
  arma::vec X_bar = arma::mean(X, 0).t();
  arma::mat Sigma = arma::cov(X);
  Sigma.diag() += 1e-6;
  arma::mat L = arma::chol(Sigma);

  // Seeded intervals
  arma::umat intervals = generate_seeded_intervals(n, beta, m_min);
  int num_intervals = intervals.n_rows;
  int k_min = intervals(num_intervals - 1, 2);

  // Define a latent vector indicating active intervals
  arma::uvec active(num_intervals, arma::fill::ones);
  std::vector<CP_Result> final_results;

#ifdef _OPENMP
  if (n_parallel > 0) omp_set_num_threads(n_parallel);
#endif

  // Layer-wise detection
  for (int k = k_min; k >= 1; --k) {
    std::vector<Candidate> candidates;

    #pragma omp parallel for
    for (int i = 0; i < num_intervals; ++i) {
      if (static_cast<int>(intervals(i, 2)) == k && active[i]) {
        int s = intervals(i, 0);
        int e = intervals(i, 1);

        arma::vec res = compute_mxPBF_cov(X, s, e, 0.0, trunc,
                                      b_I, b_L, b_R, a0);

        if (res[0] > C_cp) {
          #pragma omp critical
          {
            candidates.push_back({i, res[0], static_cast<int>(std::round(res[1])), s, e});
          }
        } else {
          active[i] = 0;
        }
      }
    }

    // Skip to the next layer if there is no candidates
    if (candidates.empty()) continue;

    // Calibrate alpha
    int omega_k = candidates[0].e - candidates[0].s;
    List calib_res = calibrate_alpha_cov(X_bar, L, omega_k, FPR_0, C_cp,
                                         N_min, N_batch, N_max, C_conv, n, p, trunc,
                                         b_I, b_L, b_R, a0);
    double alpha_k = calib_res["alpha_hat"];

    double gamma_k = std::pow(std::max(n, p), -alpha_k);
    double shift = 0.5 * std::log(gamma_k / (1.0 + gamma_k)) - 0.5 * std::log(0.5);

    std::sort(candidates.begin(), candidates.end());

    for (const auto& cand : candidates) {
      if (!active[cand.id]) continue;

      double penalized_pbf = cand.pbf_0 + shift;

      // Identify it as a change point
      if (penalized_pbf > C_cp) {
        final_results.push_back({cand.eta, alpha_k, penalized_pbf, cand.s, cand.e});

        // Remove intersecting intervals
        for (int j = 0; j < num_intervals; ++j) {
          if (active[j]) {
            int js = intervals(j, 0);
            int je = intervals(j, 1);
            if (std::max(cand.s, js) < std::min(cand.e, je)) {
              active[j] = 0;
            }
          }
        }
      }
    }
  }

  std::sort(final_results.begin(), final_results.end());

  int num_cps = final_results.size();
  arma::ivec out_cps(num_cps);
  arma::vec out_alphas(num_cps);
  arma::vec out_pbfs(num_cps);
  arma::ivec out_s(num_cps);
  arma::ivec out_e(num_cps);

  for (int i = 0; i < num_cps; ++i) {
    out_cps(i)    = final_results[i].eta + 1; // 1-based correction
    out_alphas(i) = final_results[i].alpha;
    out_pbfs(i)   = final_results[i].pbf;
    out_s(i)      = final_results[i].s + 1;
    out_e(i)      = final_results[i].e + 1;
  }

  return DataFrame::create(Named("change_point") = out_cps,
                           Named("interval_start") = out_s,
                           Named("interval_end") = out_e,
                           Named("calibrated_alpha") = out_alphas,
                           Named("mxPBF_value") = out_pbfs);
}
