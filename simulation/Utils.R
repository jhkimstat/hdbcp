reconcile_Grundy <- function(distance_points, angle_points, delta_n) {
  # If either set is empty
  if (length(angle_points) == 0) {
    return(sort(unique(distance_points)))
  }

  if (length(distance_points) == 0) {
    return(sort(unique(angle_points)))
  }

  # For each distance_point, calculate minimum distance to angle_points
  # TRUE if the distance is greater or equal than delta_n
  keep_flags <- sapply(distance_points, function(p) {
    min(abs(p - angle_points)) >= delta_n
  })

  # Filter valid distance points
  valid_distance_points <- distance_points[keep_flags]

  # Return
  final_points <- sort(unique(c(angle_points, valid_distance_points)))
  return(final_points)
}

prune_inspect <- function(changepoints, delta_n) {
  # If change point is empty
  if (is.null(changepoints) || nrow(changepoints) == 0) {
    return(integer(0))
  }

  # Sort by CUSUM
  sorted_changepoints <- changepoints[order(-changepoints[, "max.proj.cusum"]), , drop = FALSE]

  n_rows <- nrow(sorted_changepoints)
  keep <- rep(TRUE, n_rows)
  locations <- sorted_changepoints[, "location"]

  if (n_rows > 1) {
    for (i in 1:(n_rows - 1)) {
      if (keep[i]) {
        current_loc <- locations[i]
        idx_to_check <- (i + 1):n_rows
        close_idx <- idx_to_check[abs(locations[idx_to_check] - current_loc) <= delta_n]
        keep[close_idx] <- FALSE
      }
    }
  }
  # Return
  return(sort(locations[keep]))
}
