
# ==============================================================================
# High-Dimensional Bayesian Change Point Detection (hdbcp) Simulation
# ==============================================================================

# 1. Slurm Array ID
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  batch_id <- 1
} else {
  batch_id <- as.numeric(args[1])
}

# 사전에 빌드하여 설치한 hdbcp 패키지 로드
library(hdbcp)
mxPBF_mean_old <- hdbcp::mxPBF_mean
majority_rule_mxPBF_old <- hdbcp::majority_rule_mxPBF

source("hdbcp/R/data_mean_fixed.R")
source("hdbcp/R/mxPBF_mean.R")
source("hdbcp/R/Metrics.R")
Rcpp::sourceCpp("hdbcp/src/main_mean.cpp")

# Algorithm Parameters (270 combinations)
algo_grid <- expand.grid(
  beta = c(1/2, sqrt(1/2), 7/8),
  C_conv = c(0.5, 0.1, 0.05),
  C_trun = c(0, 0.125, 0.25),
  FPR0 = c(0.01, 0.05),
  m_min = c(10, 25, 50, 75, 100),
  algo_type = "proposed",
  stringsAsFactors = FALSE
)

# 2.2 Baseline Algorithm (1 combination)
baseline_grid <- data.frame(
  beta = NA, C_conv = NA, C_trun = NA, FPR0 = NA, m_min = NA,
  algo_type = "mxPBF_mean_old",
  stringsAsFactors = FALSE
)

# 2.3 Merge Grid (Total: 271 combinations)
algo_grid <- rbind(algo_grid, baseline_grid)

# Data Generating Parameters (48 combinations)
data_grid <- expand.grid(
  C_A = c(0.1, 0.5),
  delta_n = c(25, 50),
  Omega_type = c("sparse", "dense"),
  C_S = c(0.25, 0.5, 0.75, 1, 1.25, 1.5),
  stringsAsFactors = FALSE
)


if(batch_id < 1 || batch_id > nrow(algo_grid)) {
  stop("Error: batch_id is out of bounds.")
}
current_algo <- algo_grid[batch_id, ]

n <- 500
p <- 500
T_cp <- 5
n_reps <- 30
alps_val <- seq(1, 10, 0.05)
nws_val <- c(25, 60, 100)
fpr_want <- 0.05
n_sample <- 300

results_list <- list()
counter <- 1

# ==============================================================================
# Main simulation loop
# ==============================================================================


for (scen_id in 1:nrow(data_grid)) {
  current_data <- data_grid[scen_id, ]

  for (rep_id in 1:n_reps) {

    current_seed <- scen_id * 1000 + rep_id
    set.seed(current_seed)

    sim_result <- tryCatch({

      # ------------------------------------------------------------------
      # [Step 1] Data generation
      # ------------------------------------------------------------------
      sim_data <- generate_mean_data_fixed(
        n = n,
        p = p,
        T_cp = T_cp,
        C_A = current_data$C_A,
        delta_n = current_data$delta_n,
        C_S = current_data$C_S,
        omega_type = current_data$Omega_type
      )

      true_cp <- sim_data$eta

      # ------------------------------------------------------------------
      # [Step 2] Model fit
      # ------------------------------------------------------------------
      start_time <- Sys.time()

      if (current_algo$algo_type == "proposed") {
        fit <- mxPBF_mean(
          X = sim_data$X,
          beta = current_algo$beta,
          m_min = current_algo$m_min,
          FPR_0 = current_algo$FPR0,
          C_conv = current_algo$C_conv,
          trunc = current_algo$C_trun,
          n_parallel = 8
        )
        est_cp <- fit$change_point

      } else if (current_algo$algo_type == "mxPBF_mean_old") {
        # Baseline
        fit <- mxPBF_mean_old(
          given_data = sim_data$X,
          nws = nws_val,
          alps = alps_val,
          FPR_want = fpr_want,
          n_sample = n_sample,
          n_cores = 8
        )

        # 다수결 원칙으로 최종 변화점 추출
        est_cp <- majority_rule_mxPBF_old(fit)
      }

      end_time <- Sys.time()
      run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

      est_T <- length(est_cp)

      # ------------------------------------------------------------------
      # [Step 3] Evaluation
      # ------------------------------------------------------------------
      # Evaluate Change-Point Detection Performance (dH, FPSLE, FNSLE)
      metrics <- evaluate_cpd(eta_hat = est_cp, eta_true = true_cp, n = n)

      list(
        status = "Success",
        error_msg = NA,
        est_T = est_T,
        dH = metrics["dH"],
        FPSLE = metrics["FPSLE"],
        FNSLE = metrics["FNSLE"],
        run_time = run_time
      )

    }, error = function(e) {
      list(
        status = "Failed",
        error_msg = e$message,
        est_T = NA,
        dH = NA,
        FPSLE = NA,
        FNSLE = NA,
        run_time = NA
      )
    })

    # ------------------------------------------------------------------
    # [Step 4] Merge
    # ------------------------------------------------------------------
    results_list[[counter]] <- data.frame(
      batch_id = batch_id,
      algo_type = current_algo$algo_type,
      scenario_id = scen_id,
      rep_id = rep_id,
      seed = current_seed,
      beta = current_algo$beta,
      C_conv = current_algo$C_conv,
      C_trun = current_algo$C_trun,
      FPR0 = current_algo$FPR0,
      m_min = current_algo$m_min,
      C_A = current_data$C_A,
      delta_n = current_data$delta_n,
      Omega_type = current_data$Omega_type,
      C_S = current_data$C_S,
      status = sim_result$status,
      error_msg = sim_result$error_msg,
      est_T = sim_result$est_T,
      dH = sim_result$dH,
      FPSLE = sim_result$FPSLE,
      FNSLE = sim_result$FNSLE,
      run_time = sim_result$run_time,
      stringsAsFactors = FALSE
    )

    counter <- counter + 1
  }
}

# ==============================================================================
# 4. 결과 저장
# ==============================================================================
final_batch_results <- do.call(rbind, results_list)


# RDS 형태로 저장 (파일 이름: batch_001.rds ~ batch_270.rds)
save_path <- sprintf("results/batch_%03d.rds", batch_id)
saveRDS(final_batch_results, file = save_path)
