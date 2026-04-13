# ==============================================================================
# High-Dimensional Bayesian Covariance Change Point Detection Simulation
# (Sensitivity Analysis & Chunked Array Version)
# ==============================================================================

# 1. Slurm Array ID 확인
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  task_id <- 1
} else {
  task_id <- as.numeric(args[1])
}

# ------------------------------------------------------------------------------
# 패키지 및 소스 로드
# ------------------------------------------------------------------------------
library(hdbcp)
mxPBF_cov_old <- hdbcp::mxPBF_cov
majority_rule_mxPBF_old <- hdbcp::majority_rule_mxPBF

source("hdbcp/R/data_cov_fixed.R")
source("hdbcp/R/mxPBF_cov.R")
source("hdbcp/R/Metrics.R")
Rcpp::sourceCpp("hdbcp/src/main_cov.cpp")

# ==============================================================================
# 2. Grid Setup (Sensitivity Analysis)
# ==============================================================================

# 2.1 Algorithm Grid (총 64개 조합)
algo_grid <- expand.grid(
  beta = c(sqrt(1/2), 7/8),
  C_conv = c(0.05, 0.1),
  C_trun = c(0.125, 0.25),
  FPR0 = c(0.01, 0.05),
  m_min = c(5, 10, 25, 50),
  algo_type = "proposed",
  stringsAsFactors = FALSE
)

# 2.2 Baseline Algorithm (1개 조합)
baseline_grid <- data.frame(
  beta = NA, C_conv = NA, C_trun = NA, FPR0 = NA, m_min = NA,
  algo_type = "mxPBF_cov_old",
  stringsAsFactors = FALSE
)

# Merge Algorithm Grid (총 65 rows)
full_algo_grid <- rbind(algo_grid, baseline_grid)
full_algo_grid$algo_id <- 1:nrow(full_algo_grid)

# 2.3 Data Generating Grid (총 48 rows)
data_grid <- expand.grid(
  signal_type = c("rare", "many"),
  delta_n = c(25, 50),
  Sigma_type = c("sparse", "dense"),
  C_S = c(1, 3, 5, 7, 9, 11),
  stringsAsFactors = FALSE
)
data_grid$scen_id <- 1:nrow(data_grid)

# 2.4 Master Grid 생성 (65 x 48 = 3,120 rows)
master_grid <- merge(full_algo_grid, data_grid, by = NULL)

# ==============================================================================
# 3. Task / Chunking Logic
# ==============================================================================
rows_per_task <- 12      # 한 Task가 담당할 세팅(Setting) 수
chunks_per_setting <- 3  # 한 세팅(30 reps)을 3개 청크로 분할
reps_per_chunk <- 10     # 한 청크당 10번 반복

# group_id: 몇 번째 12개 세팅 묶음인지 계산
group_id <- ceiling(task_id / chunks_per_setting)
# chunk_id: 해당 묶음 내에서 몇 번째 청크(1, 2, 3)인지 계산
chunk_id <- (task_id - 1) %% chunks_per_setting + 1

# 해당 Task가 담당할 Master Grid의 인덱스 계산
setting_start_idx <- (group_id - 1) * rows_per_task + 1
setting_end_idx   <- min(group_id * rows_per_task, nrow(master_grid))

if(setting_start_idx > nrow(master_grid)) {
  stop(sprintf("Error: task_id %d is out of bounds.", task_id))
}

# 현재 청크가 처리할 반복(rep) 구간 계산 (예: 1~10, 11~20, 21~30)
rep_start <- (chunk_id - 1) * reps_per_chunk + 1
rep_end   <- chunk_id * reps_per_chunk

# 실행할 세팅 목록 추출
tasks_to_run <- master_grid[setting_start_idx:setting_end_idx, ]

# 공통 고정 파라미터
n <- 500
p <- 200
T_cp <- 5
alps_val <- seq(1, 10, 0.05)
nws_val <- c(25, 60, 100)
fpr_want <- 0.05
n_sample <- 300

# ==============================================================================
# 4. Main Simulation Loop
# ==============================================================================
if (!dir.exists("results_cov")) {
  dir.create("results_cov")
}

for (i in 1:nrow(tasks_to_run)) {
  current_task <- tasks_to_run[i, ]
  current_setting_id <- (group_id - 1) * rows_per_task + i

  results_list <- list()

  for (rep_id in rep_start:rep_end) {

    # 시드 고유성 확보 (데이터 시나리오 + 알고리즘 + 반복 횟수 조합)
    current_seed <- (current_task$scen_id * 10000) + (current_task$algo_id * 100) + rep_id
    set.seed(current_seed)

    sim_result <- tryCatch({
      # [Step 1] Data generation
      sim_data <- generate_cov_data_fixed(
        n = n, p = p, T_cp = T_cp,
        delta_n = current_task$delta_n,
        C_S = current_task$C_S,
        sigma_type = current_task$Sigma_type,
        signal_type = current_task$signal_type
      )

      true_cp <- sim_data$eta

      # [Step 2] Model fit
      start_time <- Sys.time()

      if (current_task$algo_type == "proposed") {
        fit <- mxPBF_cov(
          X = sim_data$X,
          beta = current_task$beta,
          m_min = current_task$m_min,
          FPR_0 = current_task$FPR0,
          C_conv = current_task$C_conv,
          trunc = current_task$C_trun,
          n_parallel = 8
        )
        est_cp <- fit$change_point

      } else if (current_task$algo_type == "mxPBF_cov_old") {
        fit <- mxPBF_cov_old(
          given_data = sim_data$X,
          nws = nws_val, alps = alps_val,
          FPR_want = fpr_want, n_sample = n_sample,
          n_cores = 8
        )
        est_cp <- majority_rule_mxPBF_old(fit)
      }

      end_time <- Sys.time()
      run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

      est_T <- length(est_cp)

      # [Step 3] Evaluation
      metrics <- evaluate_cpd(eta_hat = est_cp, eta_true = true_cp, n = n)

      list(
        status = "Success", error_msg = NA,
        est_T = est_T, dH = metrics["dH"],
        FPSLE = metrics["FPSLE"], FNSLE = metrics["FNSLE"], run_time = run_time
      )

    }, error = function(e) {
      list(
        status = "Failed", error_msg = e$message,
        est_T = NA, dH = NA, FPSLE = NA, FNSLE = NA, run_time = NA
      )
    })

    # [Step 4] Save Current Rep
    # 결과 리스트의 인덱스를 1부터 시작하도록 매핑
    list_idx <- rep_id - rep_start + 1

    results_list[[list_idx]] <- data.frame(
      task_id = task_id,
      setting_id = current_setting_id,
      chunk_id = chunk_id,
      algo_type = current_task$algo_type,
      scenario_id = current_task$scen_id,
      rep_id = rep_id,
      seed = current_seed,
      beta = current_task$beta,
      C_conv = current_task$C_conv,
      C_trun = current_task$C_trun,
      FPR0 = current_task$FPR0,
      m_min = current_task$m_min,
      delta_n = current_task$delta_n,
      Sigma_type = current_task$Sigma_type,
      signal_type = current_task$signal_type,
      C_S = current_task$C_S,
      status = sim_result$status,
      error_msg = sim_result$error_msg,
      est_T = sim_result$est_T,
      dH = sim_result$dH,
      FPSLE = sim_result$FPSLE,
      FNSLE = sim_result$FNSLE,
      run_time = sim_result$run_time,
      stringsAsFactors = FALSE
    )
  }

  # ============================================================================
  # 5. 결과 저장 (Setting ID와 Chunk ID를 파일명에 명시)
  # ============================================================================
  final_task_results <- do.call(rbind, results_list)
  save_path <- sprintf("results_cov/sim_S%04d_C%02d.rds", current_setting_id, chunk_id)
  saveRDS(final_task_results, file = save_path)
}
