# ==============================================================================
# High-Dimensional Bayesian Change Point Detection (hdbcp) Simulation
# ==============================================================================

# 0. Load packages and functions
rm(list=ls())
library(Rcpp)
library(RcppArmadillo)
library(changepoint.geo)
library(changepoint.np)
library(ecp)
source("hdbcp/R/data_cov_fixed.R")
source("hdbcp/R/mxPBF_cov.R")
source("hdbcp/R/Metrics.R")
Rcpp::sourceCpp("hdbcp/src/main_cov.cpp")
Rcpp::sourceCpp("hdbcp/src/Dette2022.cpp")
source("hdbcp/simulation/Utils.R")

# Parse Command Line Arguments (SLURM Task ID)
args <- commandArgs(trailingOnly = TRUE)
task_id <- ifelse(length(args) == 0, 1, as.integer(args[1]))

# 1개의 세팅을 5개의 청크로 나눔 (1 청크당 10 reps -> 총 50 reps)
chunks_per_setting <- 5  
reps_per_chunk <- 10     

setting_id <- ceiling(task_id / chunks_per_setting)
chunk_id <- (task_id - 1) %% chunks_per_setting + 1

# 해당 Task가 담당할 rep 번호 계산 (예: chunk 1은 1~10, chunk 2는 11~20)
rep_start <- (chunk_id - 1) * reps_per_chunk + 1
rep_end <- chunk_id * reps_per_chunk

# Parameter Grid Setup
n <- 500
delta_n <- 25

# 공분산 시뮬레이션 세팅
p_vec <- c(200, 500)
T_vec <- c(1, 5)
CA_vec <- c(0.1, 0.5, 1.0)
Sigma_vec <- c("sparse", "dense") 
CS_vec <- c(1, 3, 5, 7, 9, 11)

# 총 144개의 세팅 (144 * 5 = 720 Tasks)
param_grid <- expand.grid(p = p_vec, T = T_vec, CA = CA_vec, Sigma = Sigma_vec, CS = CS_vec)
current_setting <- param_grid[setting_id, ]

p_val <- current_setting$p
T_val <- current_setting$T
CA_val <- current_setting$CA
Sigma_val <- as.character(current_setting$Sigma)
CS_val <- current_setting$CS

# 폴더 사전 생성
save_dir <- "results_cov"
if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

for (rep_id in rep_start:rep_end) {
  
  save_path <- sprintf("%s/sim_set_%03d_rep_%03d.rds", save_dir, setting_id, rep_id)
  
  # 이미 존재하는 파일 스킵 (이어하기 로직 유지)
  if (file.exists(save_path)) {
    cat(sprintf("Rep %d already exists. Skipping...\n", rep_id))
    next
  }
  
  # 재현성을 위한 시드 고정
  set.seed(setting_id * 10000 + rep_id) 
  
  tryCatch({
    # Simulation
    dat <- generate_cov_data_fixed(
      n = n, p = p_val, T_cp = T_val, C_A = CA_val, 
      delta_n = delta_n, C_S = CS_val, sigma_type = Sigma_val
    )
    
    ## Proposed Method
    res_hdbcp <- mxPBF_cov(X = dat$X,
                           beta = 0.875,
                           m_min = 25,
                           FPR_0 = 0.05,
                           C_conv = 0.1,
                           trunc = 0.25,
                           n_parallel = 8)
    
    eta_hat_hdbcp <- res_hdbcp$change_point
    metric_hdbcp <- evaluate_cpd(eta_hat = eta_hat_hdbcp, eta_true = dat$eta, n = n)
    
    ## Grundy et al. method
    cp_distance <- cpt.np(distance.mapping(dat$X),
                          penalty = "MBIC", method = "PELT",
                          test.stat = "empirical_distribution", class = TRUE,
                          minseglen = delta_n, nquantiles = 4*log(n))
    cp_angle <- cpt.np(angle.mapping(dat$X),
                       penalty = "MBIC", method = "PELT",
                       test.stat = "empirical_distribution", class = TRUE,
                       minseglen = delta_n, nquantiles = 4*log(n))
    
    raw_Grundy <- reconcile_Grundy(
      distance_points = cp_distance@cpts[-length(cp_distance@cpts)],
      angle_points = cp_angle@cpts[-length(cp_angle@cpts)],
      delta_n = delta_n)
    eta_hat_Grundy <- as.numeric(unlist(raw_Grundy))
    metric_Grundy <- evaluate_cpd(eta_hat = eta_hat_Grundy, eta_true = dat$eta, n = n)
    
    ## Dette et al. method
    if (T_val == 1) {
      res_Dette <- realest(t(dat$X))
      eta_hat_Dette <- as.numeric(unlist(res_Dette))
      metric_Dette <- evaluate_cpd(eta_hat = eta_hat_Dette, eta_true = dat$eta, n = n)
    } else {
      eta_hat_Dette <- integer(0)
      metric_Dette <- c(dH = NA, FPSLE = NA, FNSLE = NA)
    }
    
    ## E-Divisive method
    res_EDivisive <- e.divisive(dat$X, R = 499, min.size = delta_n, alpha = 1)
    if (length(res_EDivisive$estimates) > 2) {
      raw_EDivisive <- res_EDivisive$estimates[2:(length(res_EDivisive$estimates) - 1)]
    } else {
      raw_EDivisive <- integer(0)
    }
    eta_hat_EDivisive <- as.numeric(unlist(raw_EDivisive))
    metric_EDivisive <- evaluate_cpd(eta_hat = eta_hat_EDivisive, eta_true = dat$eta, n = n)
    
    # 4. Save Result
    result <- list(
      setting_id = setting_id,
      rep_id = rep_id,
      setting = current_setting,
      true_eta = dat$eta,
      
      # Estimates
      est_eta_hdbcp = eta_hat_hdbcp,
      est_eta_Grundy = eta_hat_Grundy,
      est_eta_Dette = eta_hat_Dette,
      est_eta_ecp = eta_hat_EDivisive,
      
      
      # Metrics
      metric_hdbcp = metric_hdbcp,
      metric_Grundy = metric_Grundy,
      metric_Dette = metric_Dette,
      metric_ecp = metric_EDivisive
    )
    
    saveRDS(result, file = save_path)
    cat(sprintf("Rep %d successfully saved.\n", rep_id))
    
  }, error = function(e) {
    cat(sprintf("Error at Rep %d: %s\n", rep_id, e$message))
  })
  
  gc() # 메모리 초기화
}