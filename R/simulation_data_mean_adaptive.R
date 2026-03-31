#' Simulation data generating function for High-Dimensional Mean Bayesian Change Point Detection (Adaptive Signal Version)
#'
#' @param n The number of observations (time length). (Default: 500).
#' @param p The dimension of the matrix (Default: 500).
#' @param T_cp The number of true change-points to generate (Default: 10).
#' @param C_A The proportion of active dimensions where the mean changes occur (Default: 0.1).
#' @param delta_n The minimum spacing condition between consecutive change-points. It should be less than or equal to \eqn{\lfloor n / (T_{cp} + 1) \rfloor} (Default: 20).
#' @param C_S A constant controlling the base signal strength. (Default: \code{sqrt(10)}).
#' @param rho The spatial correlation parameter between -1 and 1. If 0, the covariance matrix is diagonal (Default: 0).
#' @param vanishing If \code{TRUE}, applies the vanishing signals setting where jump sizes decrease as the number of active coordinates increases (Default: \code{FALSE}).
#'
#' @return A list containing the following components:
#' \describe{
#'   \item{X}{An \code{n} by \code{p} numeric matrix representing the generated multivariate time series data.}
#'   \item{eta}{A numeric vector containing the true locations of the change-points.}
#'   \item{Sigma}{A \code{p} by \code{p} numeric matrix representing the true covariance matrix used for data generation.}
#' }
#'
#' @importFrom MASS mvrnorm
#' @importFrom stats runif rbinom
#' @export
#'
#' @examples
#' set.seed(2026)
#' # 1. Base Process
#' dat_base <- generate_mean_data_adaptive(n = 300, p = 100, T_cp = 5)
#'
#' # 2. Spatial Correlation
#' dat_spatial <- generate_mean_data_adaptive(n = 300, p = 100, T_cp = 5, rho = 0.6)
#'
#' # 3. Vanishing Signals
#' dat_vanish <- generate_mean_data_adaptive(n = 300, p = 100, T_cp = 5, vanishing = TRUE)
generate_mean_data_adaptive <- function(n = 500, p = 500, T_cp = 10, C_A = 0.1,
                              delta_n = 20, C_S = sqrt(10), rho = 0, vanishing = FALSE) {

  # Draw change point locations
  S_free <- n - (T_cp + 1) * delta_n
  if (S_free < 0) {
    stop("Invalid parameters: The minimum spacing condition (delta_n) and number of change points (T_cp) exceed the data length (n).")
  }
  eta_init <- sort(sample(1:(S_free + T_cp), T_cp))

  k <- 1:T_cp
  eta_inner <- eta_init - k + 1 + k * delta_n
  eta <- c(1, eta_inner, n + 1)

  # Generate covariance matrix
  U <- stats::runif(p, min = -2, max = 2)
  sigma_vec <- 2^U
  sd_vec <- sqrt(sigma_vec)

  if (rho == 0) {
    Sigma <- diag(sigma_vec)
  } else {
    dist_mat <- abs(outer(1:p, 1:p, "-"))
    corr_mat <- rho^dist_mat
    Sigma <- corr_mat * outer(sd_vec, sd_vec, "*")
  }

  # Select active coordinates
  p0 <- floor(C_A * p)
  A <- sample(1:p, p0)

  # Generate mean vector and data
  X <- matrix(0, nrow = n, ncol = p)
  mu_current <- rep(0, p)

  for (t in 1:(T_cp + 1)) {
    start_idx <- eta[t]
    end_idx <- eta[t+1] - 1


    if (t > 1) {
      Y <- stats::rbinom(p, 1, 0.5)
      sign_mult <- 1 - 2 * Y

      min_dist <- min(eta[t+1] - eta[t], eta[t] - eta[t-1])

      jump <- (C_S * sign_mult * sigma_vec) / sqrt(min_dist)

      if (vanishing) {
        jump <- jump / sqrt(p0)
      }

      jump[-A] <- 0
      mu_current <- mu_current + jump
    }

    n_obs <- end_idx - start_idx + 1

    if (n_obs > 0) {
      X[start_idx:end_idx, ] <- MASS::mvrnorm(n_obs, mu = mu_current, Sigma = Sigma)
    }
  }

  return(list(
    X = X,
    eta = eta_inner,
    Sigma = Sigma
  ))
}
