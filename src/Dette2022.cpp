#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat kro(const arma::mat& data) {
  int p = data.n_rows;
  int n = data.n_cols;
  int p2 = p * (p + 1) / 2;
  arma::mat X(p2, n, arma::fill::zeros);
  
  for (int k = 0; k < n; ++k) {
    int idx = 0;
    for (int j = 0; j < p; ++j) {
      for (int i = j; i < p; ++i) { // Lower triangle indices
        X(idx++, k) = data(i, k) * data(j, k);
      }
    }
  }
  return X;
}

// [[Rcpp::export]]
arma::vec reduvec(const arma::mat& T) {
  int n = T.n_cols;
  
  arma::vec Tsum = arma::sum(T, 1);
  arma::mat CU = arma::cumsum(T, 1);
  arma::mat CU1 = arma::square(CU);
  arma::mat CU2 = arma::cumsum(arma::square(T), 1);
  
  arma::mat RCU = -CU;
  RCU.each_col() += Tsum;
  arma::mat RCU1 = arma::square(RCU);
  
  arma::vec Tsq_sum = arma::sum(arma::square(T), 1);
  arma::mat RCU2 = -CU2;
  RCU2.each_col() += Tsq_sum;
  
  arma::mat MIX = CU % RCU;
  
  arma::rowvec a(n, fill::zeros), b(n, fill::zeros), c(n, fill::zeros);
  
  for(int k = 1; k <= n; ++k) {
    a(k-1) = (k == 1) ? 0.0 : (double)(n - k) / ((double)n * (k - 1) * (n - 3));
    b(k-1) = (k >= n - 1) ? 0.0 : (double)k / ((double)n * (n - k - 1) * (n - 3));
    c(k-1) = 2.0 / ((double)n * (n - 3));
  }
  
  arma::mat A = (CU1 - CU2).eval().each_row() % a;
  arma::mat B = (RCU1 - RCU2).eval().each_row() % b;
  arma::mat C = MIX.each_row() % c;
  
  // MATLAB의 A(:,[1,n-1,n])=[] 와 동일. 0-based index이므로 1부터 n-3까지 추출
  arma::mat A_sub = A.cols(1, n - 3);
  arma::mat B_sub = B.cols(1, n - 3);
  arma::mat C_sub = C.cols(1, n - 3);
  
  arma::vec T1 = arma::sum(A_sub, 1);
  arma::vec T2 = arma::sum(B_sub, 1);
  arma::vec T3 = arma::sum(C_sub, 1);
  
  return T1 + T2 - T3;
}

// [[Rcpp::export]]
arma::uvec redupos(const arma::mat& data) {
  int p = data.n_rows;
  int n = data.n_cols;
  
  arma::vec mean_vec = arma::mean(data, 1);
  arma::mat ndata = data.each_col() - mean_vec;
  
  arma::mat T = kro(ndata);
  arma::vec s = reduvec(T);
  
  int p2 = p * (p + 1) / 2;
  int n_half = n / 2;
  arma::mat Z(p2, n_half, fill::zeros);
  
  for(int k = 0; k < n_half; ++k) {
    Z.col(k) = T.col(2 * k + 1) - T.col(2 * k);
  }
  
  arma::vec d = (1.0 / std::sqrt(2.0)) * arma::stddev(Z, 0, 1); 
  
  arma::mat gen = arma::randn<arma::mat>(p2, n);
  arma::mat new_mat = gen.each_col() % d;
  
  arma::vec snew = reduvec(new_mat);
  double cr = arma::max(snew);
  
  arma::uvec ind = arma::find(s > cr);
  return ind;
}

// [[Rcpp::export]]
arma::ivec realest(arma::mat data3) {
  int n = data3.n_cols;
  int ln = 2; int rn = n - 2;
  
  arma::uvec ind = redupos(data3);
  
  if (ind.is_empty()) {
    return arma::ivec();
  }
  
  arma::vec mean_vec = arma::mean(data3, 1);
  arma::mat data = data3.each_col() - mean_vec;
  
  arma::mat T = kro(data);
  arma::mat X = T.rows(ind); 
  
  arma::rowvec estim(n, fill::zeros);
  
  for (int k = ln; k <= rn; ++k) {
    arma::mat dk1 = X.cols(0, k - 1);
    arma::mat dk2 = X.cols(k, n - 1);
    
    arma::vec s1 = arma::sum(dk1, 1);
    double sum_dk1_sq = arma::accu(arma::square(dk1));
    double Q1 = (1.0 / (k * (k - 1))) * (arma::dot(s1, s1) - sum_dk1_sq);
    
    arma::vec s2 = arma::sum(dk2, 1);
    double sum_dk2_sq = arma::accu(arma::square(dk2));
    double n_minus_k = n - k;
    double Q2 = (1.0 / (n_minus_k * (n_minus_k - 1))) * (arma::dot(s2, s2) - sum_dk2_sq);
    
    double Q3 = (1.0 / (k * n_minus_k)) * arma::dot(s1, s2);
    
    double Q = Q1 + Q2 - 2.0 * Q3;
    
    estim(k - 1) = ((double)k * (k - 1) * n_minus_k * (n_minus_k - 1)) / std::pow((double)n, 4.0) * Q;
  }
  
  arma::rowvec estimcut = estim.subvec(ln - 1, rn - 1);
  arma::uword I = estimcut.index_max(); 

  int khat = (double)I + ln; // 0-based index correction
  return arma::ivec({khat});
}