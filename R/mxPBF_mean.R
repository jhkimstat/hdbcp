#' High-Dimensional Bayesian Change Point Detection (Mean shift version)
#'
#' @param X A numeric matrix or data.frame of shape n x p.
#' @param rho Scale parameter for interval generation (Default: 1 / sqrt(2)). Must be strictly between 0 and 1.
#' @param m_min Minimum search interval length (Default: ceiling(8 * log(max(n, p))) + 1).
#' @param FPR_0 Target False Positive Rate (Default: 0.05).
#' @param C_cp Threshold for change point detection (Default: log(10)).
#' @param N_min Initial number of simulations for alpha calibration (Default: 150).
#' @param N_batch Additional batch size for calibration simulations (Default: 50).
#' @param N_max Maximum number of simulations for calibration (Default: 500).
#' @param C_conv Early stopping criterion for calibration convergence (Default: 0.01).
#' @param trunc Truncation proportion at the boundaries during the local search (Default: 0.25).
#' @param n_parallel Number of threads for OpenMP parallel execution (Default: 1).
#'
#' @return A data.frame containing the estimated change point locations, calibrated alphas, mxPBF values, and the search intervals.
#' @export
#'
#' @examples
#' \dontrun{
#' # 1. Generate a toy data
#' set.seed(2026)
#' n <- 1000
#' p <- 20
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' X[501:n, 1:5] <- X[501:n, 1:5] + 1.5
#'
#' # 2. Run the algorithm
#' result <- mxPBF_mean(X)
#'
#' # 3. View the results
#' print(result)
#' }
mxPBF_mean <- function(X,
                       rho = 1/sqrt(2),
                       m_min = ceiling(8 * log(max(n, p))) + 1,
                       FPR_0 = 0.05,
                       C_cp = log(10),
                       N_min = 150,
                       N_batch = 50,
                       N_max = 500,
                       C_conv = 0.01,
                       trunc = 0.25,
                       n_parallel = 1) {
  # 1. Data (X) Integrity Check
  if (!is.matrix(X) && !is.data.frame(X)) {
    stop("Error: 'X' must be a matrix or a data.frame.")
  }

  X <- as.matrix(X)

  if (!is.numeric(X)) {
    stop("Error: 'X' must contain only numeric data.")
  }

  if (any(is.na(X))) {
    stop("Error: 'X' contains missing values (NA). Please impute or preprocess the data before running the algorithm.")
  }

  n <- nrow(X)

  # 2. Hyperparameter Range Validation
  if (rho <= 0 || rho >= 1) {
    stop("Error: 'rho' must be strictly between 0 and 1 (e.g., 1/sqrt(2) or 0.5).")
  }

  if (m_min < 2 || m_min >= n) {
    stop("Error: 'm_min' must be at least 2 and strictly less than the total data length (n).")
  }

  if (FPR_0 <= 0 || FPR_0 >= 1) {
    stop("Error: 'FPR_0' must be a probability strictly between 0 and 1 (e.g., 0.05).")
  }

  if (C_cp <= 0) {
    stop("Error: 'C_cp' must be greater than 0 (log(10) or higher is recommended).")
  }

  if (trunc < 0 || trunc >= 0.5) {
    stop("Error: 'trunc' proportion must be between 0 (inclusive) and 0.5 (exclusive).")
  }

  if (N_min <= 0 || N_batch <= 0 || N_max < N_min) {
    stop("Error: N_min and N_batch must be >= 1, and N_max must be >= N_min.")
  }

  if (n_parallel < 1) {
    stop("Error: 'n_parallel' (number of threads) must be at least 1.")
  }

  result <- main_mean(
    X = X,
    rho = as.numeric(rho),
    m_min = as.numeric(m_min),
    FPR_0 = as.numeric(FPR_0),
    C_cp = as.numeric(C_cp),
    N_min = as.integer(N_min),
    N_batch = as.integer(N_batch),
    N_max = as.integer(N_max),
    C_conv = as.numeric(C_conv),
    trunc = as.numeric(trunc),
    n_parallel = as.integer(n_parallel)
  )

  return(result)
}
