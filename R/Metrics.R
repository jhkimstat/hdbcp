#' Evaluate Change-Point Detection Performance
#'
#' @description
#' Calculates the Hausdorff distance (dH), False Positive Sensitive Location Error (FPSLE),
#' and False Negative Sensitive Location Error (FNSLE) between estimated and true change-points.
#'
#' @param eta_hat A numeric vector of estimated change-point locations.
#' @param eta_true A numeric vector of true change-point locations.
#' @param n The number of observations (time length).
#'
#' @return A named numeric vector containing dH, FPSLE, and FNSLE.
#' @export
#'
#' @examples
#' # n = 300, true CPs at 100, 200
#' eta_true <- c(100, 200)
#' # estimated CPs at 98, 205, 250 (one false positive)
#' eta_hat <- c(98, 205, 250)
#'
#' metrics <- evaluate_cpd(eta_hat, eta_true, n = 300)
#' print(metrics)
evaluate_cpd <- function(eta_hat, eta_true, n) {

  # Sort and add trivial change points
  eta_hat_full <- c(1, sort(eta_hat), n + 1)
  eta_true_full <- c(1, sort(eta_true), n + 1)

  # Metric 1: Hausdorff Distance (dH)
  # max_j min_i |eta_j - hat_eta_i|
  dist_true_to_hat <- max(sapply(eta_true_full, function(t) min(abs(t - eta_hat_full))))
  # max_j min_i |hat_eta_j - eta_i|
  dist_hat_to_true <- max(sapply(eta_hat_full, function(t) min(abs(t - eta_true_full))))

  dH <- dist_true_to_hat + dist_hat_to_true

  # Helper Function: Calculate directional FPSLE
  calc_fpsle_directed <- function(est_full, true_full) {
    n_segments <- length(est_full) - 1
    sum_error <- 0

    for (j in 1:n_segments) {
      # Midpoint
      M_j <- (est_full[j] + est_full[j+1]) / 2

      # Map to the true segment
      i_j <- findInterval(M_j, true_full, left.open = TRUE)

      if (i_j == 0) i_j <- 1
      if (i_j > length(true_full) - 1) i_j <- length(true_full) - 1

      # Calculate errors
      err <- abs(est_full[j] - true_full[i_j]) +
        abs(est_full[j+1] - true_full[i_j+1])

      sum_error <- sum_error + err
    }

    # Calculate the average
    return(sum_error / (2 * n_segments))
  }

  # FPSLE
  fpsle <- calc_fpsle_directed(est_full = eta_hat_full, true_full = eta_true_full)

  # FNSLE
  fnsle <- calc_fpsle_directed(est_full = eta_true_full, true_full = eta_hat_full)

  return(c(dH = dH, FPSLE = fpsle, FNSLE = fnsle))
}
