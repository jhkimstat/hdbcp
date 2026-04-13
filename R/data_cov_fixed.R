#' Simulation data generating function for High-Dimensional Bayesian Covariance Change Point Detection
#'
#' @param n The number of observations (time length). (Default: 500).
#' @param p The dimension of the matrix (Default: 500).
#' @param T_cp The number of true change-points to generate (Default: 5).
#' @param delta_n The minimum spacing condition between consecutive change-points. It should be less than or equal to \eqn{\lfloor n / (T_{cp} + 1) \rfloor} (Default: 20).
#' @param C_S A constant controlling the magnitude of the signals. (Default: 3).
#' @param sigma_type The base covariance structure. Either \code{"sparse"} or \code{"dense"}. Default is \code{"sparse"}.
#' @param signal_type The type of covariance change signals. Either \code{"rare"} (only 5 off-diagonal entries change) or \code{"many"} (dense rank-1 update). Default is \code{"rare"}.
#'
#' @return A list containing the following components:
#' \describe{
#'   \item{\code{X}}{An \code{n} by \code{p} numeric matrix of the generated data.}
#'   \item{\code{eta}}{A numeric vector of length \code{T_cp} containing the true change point locations.}
#'   \item{\code{Sigma_list}}{A list of length \code{T_cp + 1} containing the true covariance matrices for each segment.}
#' }
#'
#' @importFrom MASS mvrnorm
#' @export
#'
#' @examples
#' # Generate data with rare signals (Sparse base, Rare changes)
#' set.seed(42)
#' sim_rare <- generate_cov_data_fixed(p = 200, C_S = 1.0, sigma_type = "sparse", signal_type = "rare")
#'
#' # Generate data with many signals (Dense base, Many changes)
#' sim_many <- generate_cov_data_fixed(p = 200, C_S = 1.0, sigma_type = "dense", signal_type = "many")
generate_cov_data_fixed <- function(n = 500, p = 500, T_cp = 5,
                                    delta_n = 20, C_S = 3,
                                    sigma_type = c("sparse", "dense"),
                                    signal_type = c("rare", "many")) {

  sigma_type <- match.arg(sigma_type)
  signal_type <- match.arg(signal_type)

  # 1. Change Point Location Generation
  S_free <- n - (T_cp + 1) * delta_n
  if (S_free < 0) {
    stop("Invalid parameters: The minimum spacing condition (delta_n) and number of change points (T_cp) exceed the data length (n).")
  }

  eta_init <- sort(sample(1:(S_free + T_cp), T_cp))
  k <- 1:T_cp
  eta_inner <- eta_init - k + 1 + k * delta_n
  eta <- c(1, eta_inner, n + 1)

  # 2. Base Covariance Matrix Construction
  if (sigma_type == "sparse") {
    Delta1 <- matrix(0, p, p)
    lower_idx <- which(lower.tri(Delta1, diag = FALSE))

    selected_idx <- sample(lower_idx, floor(length(lower_idx) * 0.05))
    Delta1[selected_idx] <- 0.5
    Delta1 <- Delta1 + t(Delta1) # Symmetrize

    min_eig_D1 <- min(eigen(Delta1, symmetric = TRUE, only.values = TRUE)$values)
    Delta <- Delta1 + (abs(min_eig_D1) + 0.05) * diag(p)

    d <- runif(p, 0.5, 2.5)
    D_half <- diag(sqrt(d))
    Sigma_base <- D_half %*% Delta %*% D_half

  } else { # dense
    o_vec <- runif(p, 1, 5)
    O_mat <- diag(o_vec)

    # Construct Delta
    idx_seq <- 1:p
    diff_mat <- abs(outer(idx_seq, idx_seq, "-"))
    sum_mat <- outer(idx_seq, idx_seq, "+")

    Delta <- ((-1)^sum_mat) * (0.4^(diff_mat^(1/10)))
    Sigma_base <- O_mat %*% Delta %*% O_mat
  }

  # 3. Recursive Signal Construction
  Sigma_list <- vector("list", T_cp + 1)
  Sigma_list[[1]] <- Sigma_base

  lower_idx_no_diag <- which(lower.tri(matrix(0, p, p), diag = FALSE))

  for (t in 1:T_cp) {
    U <- matrix(0, p, p)

    if (signal_type == "rare") {
      n_rare <- 5
      selected_idx <- sample(lower_idx_no_diag, n_rare)

      U[selected_idx] <- runif(n_rare, 0, C_S)
      U <- U + t(U)

    } else {
      u_vec <- runif(p, 0, sqrt(C_S))
      U <- u_vec %*% t(u_vec)
    }
    Sigma_list[[t + 1]] <- Sigma_list[[t]] + U
  }

  # 4. Global Positive Definite Correction
  min_eigs <- sapply(Sigma_list, function(S) {
    min(eigen(S, symmetric = TRUE, only.values = TRUE)$values)
  })
  lambda_star <- min(min_eigs)

  if (lambda_star <= 0) {
    correction <- (-lambda_star + 0.05) * diag(p)
    Sigma_list <- lapply(Sigma_list, function(S) S + correction)
  }

  # 5. Data Generation
  X <- matrix(0, nrow = n, ncol = p)

  for (t in 0:T_cp) {
    start_idx <- eta[t + 1]
    end_idx <- eta[t + 2] - 1
    n_seg <- end_idx - start_idx + 1
    X[start_idx:end_idx, ] <- MASS::mvrnorm(n_seg, mu = rep(0, p), Sigma = Sigma_list[[t + 1]])
  }

  return(list(
    X = X,
    eta = eta_inner,
    Sigma_list = Sigma_list
  ))
}
