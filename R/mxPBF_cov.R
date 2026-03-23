#' High-Dimensional Bayesian Change Point Detection (Covariance shift version)
#'
#' @param X A numeric matrix or data.frame of shape n x p.
#' @param beta Scale parameter for interval generation (Default: 1 / sqrt(2)). Must be strictly between 0 and 1.
#' @param m_min Minimum search interval length (Default: 30).
#' @param FPR_0 Target False Positive Rate (Default: 0.05).
#' @param C_cp Threshold for change point detection (Default: log(10)).
#' @param N_min Initial number of simulations for alpha calibration (Default: 150).
#' @param N_batch Additional batch size for calibration simulations (Default: 50).
#' @param N_max Maximum number of simulations for calibration (Default: 500).
#' @param C_conv Early stopping criterion for calibration convergence (Default: 0.01).
#' @param trunc Truncation proportion at the boundaries during the local search (Default: 0.25).
#' @param b_I Hyperparameter for tau (Default: 0.1).
#' @param b_L Hyperparameter for left segment tau (Default: 0.1).
#' @param b_R Hyperparameter for right segment tau (Default: 0.1).
#' @param a0 Hyperparameter for tau (Default: 0.1).
#' @param n_parallel Number of threads for OpenMP parallel execution (Default: 1).
#'
#' @return A data.frame containing the estimated change point locations, calibrated alphas, mxPBF values, and the search intervals.
#' @export
#'
#' @examples
#' \dontrun{
#' # 1. Generate toy covariance change data
#' set.seed(2026)
#' dat <- generate_cov_data(n = 1000, p = 50, T_cp = 1, C_prop = 0.2)
#' X <- dat$X
#'
#' # 2. Run the covariance change point detection algorithm
#' result <- mxPBF_cov(X)
#'
#' # 3. View the results
#' print(result)
#' }
mxPBF_cov <- function(X,
                      beta = 1/sqrt(2),
                      m_min = 30,
                      FPR_0 = 0.05,
                      C_cp = log(10),
                      N_min = 150,
                      N_batch = 50,
                      N_max = 500,
                      C_conv = 0.01,
                      trunc = 0.25,
                      b_I = 0.1,
                      b_L = 0.1,
                      b_R = 0.1,
                      a0 = 0.1,
                      n_parallel = 1) {
  
  # 1. Data (X) Integrity Check
  if (!is.matrix(X) && !is.data.frame(X)) {
    stop("Error: 'X' must be a matrix or a data.frame.")
  }
  
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  
  if (!is.numeric(X)) {
    stop("Error: 'X' must contain only numeric data.")
  }
  
  if (any(is.na(X))) {
    stop("Error: 'X' contains missing values (NA). Please impute or preprocess the data.")
  }
  
  # 2. Hyperparameter Range Validation
  
  if (beta <= 0 || beta >= 1) {
    stop("Error: 'beta' must be strictly between 0 and 1.")
  }
  
  if (m_min < 2 || m_min >= n) {
    stop("Error: 'm_min' must be at least 2 and strictly less than n.")
  }
  
  if (FPR_0 <= 0 || FPR_0 >= 1) {
    stop("Error: 'FPR_0' must be a probability between 0 and 1.")
  }
  
  if (C_cp <= 0) {
    stop("Error: 'C_cp' must be greater than 0.")
  }
  
  if (trunc < 0 || trunc >= 0.5) {
    stop("Error: 'trunc' proportion must be between 0 and 0.5.")
  }
  
  if (any(c(b_I, b_L, b_R, a0) <= 0)) {
    stop("Error: Hyperparameters 'b_I', 'b_L', 'b_R', and 'a0' must be positive.")
  }
  
  if (N_min <= 0 || N_batch <= 0 || N_max < N_min) {
    stop("Error: N_min, N_batch must be >= 1, and N_max >= N_min.")
  }
  
  if (n_parallel < 1) {
    stop("Error: 'n_parallel' must be at least 1.")
  }
  
  # 3. Call Core C++ Backend (main_cov)
  # Note: The backend for covariance uses different likelihood logic (e.g., Wishart or Matrix Normal)
  result <- main_cov(
    X = X,
    beta = as.numeric(beta),
    m_min = as.numeric(m_min),
    FPR_0 = as.numeric(FPR_0),
    C_cp = as.numeric(C_cp),
    N_min = as.integer(N_min),
    N_batch = as.integer(N_batch),
    N_max = as.integer(N_max),
    C_conv = as.numeric(C_conv),
    trunc = as.numeric(trunc),
    b_I = as.numeric(b_I),     
    b_L = as.numeric(b_L),     
    b_R = as.numeric(b_R),     
    a0 = as.numeric(a0),
    n_parallel = as.integer(n_parallel)
  )
  
  return(result)
}
