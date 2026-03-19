#include <RcppArmadillo.h>
#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
void precompute_cumsums(const arma::mat& X, int n, int p, arma::mat& S, arma::mat& Q) {
  S.rows(1, n) = arma::cumsum(X, 0);
  Q.rows(1, n) = arma::cumsum(arma::square(X), 0);
}
