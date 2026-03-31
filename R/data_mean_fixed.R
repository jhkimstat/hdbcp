#' Simulation data generating function for High-Dimensional Mean Bayesian Change Point Detection (Fixed Signal Version)
#'
#' @description
#' Generates multivariate time series data with multiple change points in the mean structure.
#' The data is generated based on a specified precision matrix (sparse or dense) and
#' the mean vector follows a random walk at each change point for a subset of active dimensions.
#'
#' @param n The number of observations (time length). (Default: 500).
#' @param p The dimension of the matrix (Default: 500).
#' @param T_cp The number of true change-points to generate (Default: 5).
#' @param C_A The proportion of active dimensions where the mean changes occur (Default: 0.1).
#' @param delta_n The minimum spacing condition between consecutive change-points. It should be less than or equal to \eqn{\lfloor n / (T_{cp} + 1) \rfloor} (Default: 20).
#' @param C_S A constant controlling the base signal strength. (Default: 1).
#' @param omega_type The structure of the true precision matrix. Either \code{"sparse"} or \code{"dense"}. (Default: \code{"sparse"}).
#'
#' @return A list containing the following components:
#' \describe{
#'   \item{\code{X}}{An \code{n} by \code{p} numeric matrix of the generated data.}
#'   \item{\code{eta}}{A numeric vector of length \code{T_cp} containing the true change point locations.}
#'   \item{\code{Mu}}{An \code{n} by \code{p} numeric matrix representing the true mean structure for all observations.}
#'   \item{\code{Sigma}}{A \code{p} by \code{p} numeric matrix representing the true covariance matrix.}
#' }
#'
#' @importFrom MASS mvrnorm
#' @export
#'
#' @examples
#' # Generate data with sparse precision matrix and 5 change points
#' set.seed(123)
#' sim_data <- generate_mean_data_fixed(
#'   n = 500, p = 500, T_cp = 5, C_A = 0.1, delta_n = 20, C_S = 1.0, omega_type = "sparse"
#' )
#'
#' # True change point locations
#' print(sim_data$eta)
#'
#' # Check the dimension of generated data
#' dim(sim_data$X)
generate_mean_data_fixed <- function(n = 500, p = 500, T_cp = 5, C_A = 0.1,
                                     delta_n = 20, C_S = 1.0,
                                     omega_type = c("sparse", "dense")) {

  omega_type <- match.arg(omega_type)

  # Change Point Location Generation
  S_free <- n - (T_cp + 1) * delta_n
  if (S_free < 0) {
    stop("Invalid parameters: The minimum spacing condition (delta_n) and number of change points (T_cp) exceed the data length (n).")
  }

  eta_init <- sort(sample(1:(S_free + T_cp), T_cp))
  k <- 1:T_cp
  eta_inner <- eta_init - k + 1 + k * delta_n

  # Boundary points: eta_0 = 1, eta_{T+1} = n + 1
  eta <- c(1, eta_inner, n + 1)

  # Precision and Covariance Matrix Construction
  prob <- if (omega_type == "sparse") 0.01 else 0.40
  Omega <- matrix(0, p, p)

  upper_idx <- which(upper.tri(Omega))
  selected_idx <- sample(upper_idx, size = floor((p^2) * prob / 2))

  Omega[selected_idx] <- 0.3
  Omega <- Omega + t(Omega) # Symmetrize

  # Positive Definite Correction
  eig_vals <- eigen(Omega, symmetric = TRUE, only.values = TRUE)$values
  min_eig <- min(eig_vals)

  if (min_eig <= 0) {
    Omega <- Omega + (-min_eig + 0.1^3) * diag(p)
  }

  Sigma <- solve(Omega)

  # Active Coordinate Selection (A)
  p_A <- floor(p * C_A)
  A <- sample(1:p, size = p_A, replace = FALSE)

  # Mean Structure Construction
  Mu <- matrix(0, nrow = n, ncol = p)
  mu_current <- rep(0, p) # Initializes mu^(0) = 0

  for (t in 0:T_cp) {
    start_idx <- eta[t + 1]
    end_idx <- eta[t + 2] - 1

    if (t > 0) {
      xi <- rbinom(p_A, size = 1, prob = 0.5)
      mu_current[A] <- mu_current[A] + C_S * (1 - 2 * xi)
    }

    Mu[start_idx:end_idx, ] <- matrix(rep(mu_current, end_idx - start_idx + 1),
                                      byrow = TRUE, ncol = p)
  }

  # Data Generation
  X_zero_mean <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  X <- X_zero_mean + Mu

  return(list(
    X = X,
    eta = eta_inner,
    Mu = Mu,
    Sigma = Sigma
  ))
}
