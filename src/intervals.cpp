#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::umat generate_seeded_intervals(int n, double rho, double m_min) {
  int k_min = std::floor(1.0 + std::log(n / m_min) / std::log(1.0 / rho));

  // Number of total intervals
  int total_intervals = 0;
  for (int k = 1; k <= k_min; ++k) {
    total_intervals += 2 * std::ceil(std::pow(1.0 / rho, k - 1)) - 1;
  }

  arma::umat seeded(total_intervals, 3);
  int count = 0;

  for (int k = 1; k <= k_min; ++k) {
    int nu_k = 2 * std::ceil(std::pow(1.0 / rho, k - 1)) - 1;
    double omega_k = n * std::pow(rho, k - 1);

    if (nu_k == 1) {
      seeded(count, 0) = 0;
      seeded(count, 1) = n;
      seeded(count, 2) = k;
      count++;
    } else {
      double xi_k = (n - omega_k) / (nu_k - 1.0);
      for (int h = 1; h <= nu_k; ++h) {
        int s = std::floor((h - 1) * xi_k);
        int e = std::ceil((h - 1) * xi_k + omega_k);

        if (e > n) e = n;

        seeded(count, 0) = s;
        seeded(count, 1) = e;
        seeded(count, 2) = k;
        count++;
      }
    }
  }

  return seeded;
}
