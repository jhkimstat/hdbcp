#' Simulation data generating function for High-Dimensional Bayesian Covariance Change Point Detection (Adaptive Signal Version)
#'
#' @param n The number of observations (time length). (Default: 500).
#' @param p The dimension of the matrix (Default: 500).
#' @param T_cp The number of true change-points to generate (Default: 10).
#' @param C_A The proportion of active off-diagonal dimensions where the covariance changes occur (Default: 0.1).
#' @param delta_n The minimum spacing condition between consecutive change-points. It should be less than or equal to \eqn{\lfloor n / (T_{cp} + 1) \rfloor} (Default: 20).
#' @param C_S A constant controlling the base signal strength. (Default: \code{sqrt(10)}).
#' @param rho The spatial correlation parameter between -1 and 1. If 0, the initial covariance matrix is diagonal (Default: 0).
#' @param vanishing If \code{TRUE}, applies the vanishing signals setting where jump sizes decrease as the number of active coordinates increases (Default: \code{FALSE}).
#'
#' @return A list containing the following components:
#' \describe{
#'   \item{X}{An \code{n} by \code{p} numeric matrix representing the generated multivariate time series data.}
#'   \item{eta}{A numeric vector containing the true locations of the change-points.}
#' }
#'
#' @importFrom MASS mvrnorm
#' @importFrom stats runif rbinom
#' @export
#'
#' @examples
#' set.seed(2026)
#' # 1. Base Process
#' dat_base <- generate_cov_data_adaptive(n = 300, p = 100, T_cp = 5)
#'
#' # 2. Spatial Correlation
#' dat_spatial <- generate_cov_data_adaptive(n = 300, p = 100, T_cp = 5, rho = 0.6)
#'
#' # 3. Vanishing Signals
#' dat_vanish <- generate_cov_data_adaptive(n = 300, p = 100, T_cp = 5, vanishing = TRUE)
generate_cov_data_adaptive <- function(n = 500, p = 500, T_cp = 10, C_A = 0.1,
                              delta_n = 20, C_S = sqrt(10), rho = 0, vanishing = FALSE) {

  # Draw change point locations
  S_free <- n - (T_cp + 1) * delta_n
  eta_init <- sort(sample(1:(S_free + T_cp), T_cp))

  k <- 1:T_cp
  eta_inner <- eta_init - k + 1 + k * delta_n
  eta <- c(1, eta_inner, n + 1)

  # Generate inital covariance matrix
  U <- stats::runif(p, min = -2, max = 2)
  sigma_vec <- 2^U
  sd_vec <- sqrt(sigma_vec)

  if (rho == 0) {
    Sigma_tilde_0 <- diag(sigma_vec)
  } else {
    dist_mat <- abs(outer(1:p, 1:p, "-"))
    corr_mat <- rho^dist_mat
    Sigma_tilde_0 <- corr_mat * outer(sd_vec, sd_vec, "*")
  }

  # Select active off-diagonal coordinates (lower triangular part)
  num_lower <- p * (p - 1) / 2
  s0 <- floor(C_A * num_lower)

  lower_indices <- which(lower.tri(Sigma_tilde_0), arr.ind = TRUE)
  active_idx <- sample(1:nrow(lower_indices), s0)
  A <- lower_indices[active_idx, , drop = FALSE] # Matrix of size (s0 x 2)

  # Generate sequential jumps
  Sigma_tilde_list <- list()
  Sigma_tilde_list[[1]] <- Sigma_tilde_0

  for (t in 1:T_cp) {
    min_dist <- min(eta[t+1] - eta[t], eta[t] - eta[t-1])
    Delta_t <- matrix(0, nrow = p, ncol = p)

    if (s0 > 0) {
      Y <- stats::rbinom(s0, 1, 0.5)
      sign_mult <- 1 - 2 * Y

      # Calculate jump sizes matching the scale invariance logic
      jump_vals <- (C_S * sign_mult * sd_vec[A[, 1]] * sd_vec[A[, 2]]) / sqrt(min_dist)

      if (vanishing) {
        jump_vals <- jump_vals / sqrt(s0)
      }

      # Assign values to lower and upper triangular parts to maintain symmetry
      Delta_t[A] <- jump_vals
      Delta_t[A[, c(2, 1)]] <- jump_vals
    }

    Sigma_tilde_list[[t+1]] <- Sigma_tilde_list[[t]] + Delta_t
  }

  # Global PD Adjustment
  # Find the minimum eigenvalue across all temporal covariance matrices
  min_eig_vals <- sapply(Sigma_tilde_list, function(M) {
    min(eigen(M, symmetric = TRUE, only.values = TRUE)$values)
  })
  lambda_min_star <- min(min_eig_vals)

  adjustment <- max(0, -lambda_min_star + 0.05)

  Sigma_list <- lapply(Sigma_tilde_list, function(M) {
    M + diag(adjustment, p)
  })

  # Generate Data
  X <- matrix(0, nrow = n, ncol = p)

  for (t in 1:(T_cp + 1)) {
    start_idx <- eta[t]
    end_idx <- eta[t+1] - 1
    n_obs <- end_idx - start_idx + 1

    if (n_obs > 0) {
      # Mean is fixed at 0 for covariance change point detection
      X[start_idx:end_idx, ] <- MASS::mvrnorm(n_obs, mu = rep(0, p), Sigma = Sigma_list[[t]])
    }
  }

  return(list(
    X = X,
    eta = eta_inner
  ))
}
