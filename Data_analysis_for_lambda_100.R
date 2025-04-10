# Load required libraries
library(MASS)
library(rstiefel)
library(ManifoldOptim)
library(Matrix)
library(progress)
library(msm)
library(readr)


# Step 1: Read adjusted matrices
adjusted_files <- list.files(path = "data", pattern = "^sampled_.*\\.csv",
    full.names = TRUE
)

new_matrices <- lapply(adjusted_files, function(file) {
    df <- read_csv(file, show_col_types = FALSE)
    list(data = as.matrix(df[, -ncol(df)]), labels = df[[ncol(df)]])
    # Separate data and labels
})

# Step 2: Set parameters
n_points <- 579  # Number of samples (n_min)
dim_u <- 5       # PCA dimension
n_samples <- length(new_matrices)  # Number of groups (files)

# Verify new_matrices length
if (n_samples < 1) {
    stop("No adjusted matrices were found.")
}

# Step 3: Initialize y_data
y_data <- array(0, dim = c(n_points, dim_u, n_samples))  # 638 x 5 x n_samples
labels_list <- vector("list", n_samples)  # Store labels

# Step 4: Populate y_data
for (i in seq_along(new_matrices)) {
    current_matrix <- new_matrices[[i]]$data
    current_labels <- new_matrices[[i]]$labels

    # Verify matrix dimensions (638 rows x 5 columns)
    if (!all(dim(current_matrix) == c(n_points, dim_u))) {
        stop(paste(
            "The dimensions of adjusted matrix", adjusted_files[i],
            "are not 638 x 5."
        ))
    }

    # Row-wise scaling of the matrix
    scaled_matrix <- t(apply(current_matrix, 1, scale))  # Standardize each row

    # Populate y_data
    y_data[,,i] <- scaled_matrix/100

    # Store labels
    labels_list[[i]] <- current_labels
}

# Step 5: Output results
cat("The dimensions of y_data are:", dim(y_data), "\n")
cat("Labels for each group are stored in labels_list.\n")


# # Function to perform optimization over the Stiefel manifold
# optimize_u <- function(u_init, A, C, max_iter = 100, tol = 1e-6, alpha = 0.01) {
#   u <- u_init
#   for (iter in 1:max_iter) {
#     # Compute the gradient
#     grad_f <- 2 * A %*% u + C  # m x d matrix
    
#     # Compute the symmetric part
#     sym_part <- (t(u) %*% grad_f + t(grad_f) %*% u) / 2  # d x d matrix
    
#     # Project the gradient onto the tangent space
#     grad_proj <- grad_f - u %*% sym_part  # m x d matrix
    
#     # Update u using a step size alpha
#     u_new <- u + alpha * grad_proj  # m x d matrix
    
#     # Re-orthonormalize u_new (retraction onto the Stiefel manifold)
#     svd_u <- svd(u_new)
#     u_new <- svd_u$u %*% t(svd_u$v)
    
#     # Check for convergence
#     delta_u <- norm(u_new - u, type = "F")
#     if (delta_u < tol) {
#       break
#     }
    
#     # Update u
#     u <- u_new
#   }
#   return(u)
# }

# # Set parameters
# alpha0 <- 2
# beta0 <- 1
# lambda <- 100
# n_iter <- 100000  # Number of iterations

# # Gibbs sampling initialization
# n <- n_samples  # Number of samples
# m <- n_points   # Rows of each y_i
# d <- dim_u      # Columns of each y_i

# S <- 1
# sigma2 <- 1
# u <- rustiefel(579,5)

# # Initialize R_matrices for Gibbs sampling
# R_matrices <- array(0, dim = c(d, d, n))  # Store estimated R_i for each sample
# for (i in 1:n) {
#   R_matrices[,,i] <- diag(d)  # Initialize as identity matrices
# }

# # Store samples for diagnostics
# S_samples <- numeric(n_iter)
# sigma2_samples <- numeric(n_iter)
# u_samples <- array(0, dim = c(m, d, n_iter))
# R_samples <- array(0, dim = c(d, d, n, n_iter))

# # Initialize progress bar
# pb <- progress_bar$new(
#   format = "Iteration :current/:total [:bar] Elapsed: :elapsed ETA: :eta",
#   total = n_iter,
#   clear = FALSE,
#   width = 60
# )

# # Gibbs sampling process
# for (iter in 1:n_iter) {
  
#   pb$tick()  # Update the progress bar
  
#   # 1. Update S
#   numerator_S <- 0
#   denominator_S <- 0
#   for (i in 1:n) {
#     numerator_S <- numerator_S + tr(t(u) %*% y_data[,,i] %*% R_matrices[,,i])
#     denominator_S <- denominator_S + tr(t(y_data[,,i]) %*% y_data[,,i])
#   }
#   S_mean <- numerator_S / denominator_S
#   S_var <- sigma2 / denominator_S
#   # Sample S from a truncated normal distribution with lower bound 0
#   S <- rtnorm(1, mean = S_mean, sd = sqrt(S_var), lower = 0, upper = Inf)
  
  
#   # 2. Update sigma^2
#   sigma2_shape <- alpha0 + n * m * d / 2
#   sigma2_scale <- beta0
#   for (i in 1:n) {
#     residual <- S * y_data[,,i] %*% t(R_matrices[,,i]) - u
#     sigma2_scale <- sigma2_scale + 0.5 * sum(residual^2)
#   }
#   sigma2 <- 1 / rgamma(1, shape = sigma2_shape, rate = sigma2_scale)
  
#   # # 3. Update each R_i
#   # for (i in 1:n) {
#   #   yi <- y_data[,,i]
#   #   # Compute the concentration matrix C for matrix Fisher distribution
#   #   C <- (S / sigma2) * t(yi) %*% u + 8 * lambda * t(yi) %*% yi %*% t(yi) %*% u
#   #   # Sample R_i^T from the matrix Fisher distribution
#   #   # Note: Ensure that rmf.matrix.gibbs() function is properly defined or use an appropriate sampling method
#   #   R_i_transpose <- rmf.matrix.gibbs(C, t(R_matrices[,,i]))
#   #   R_matrices[,,i] <- t(R_i_transpose)
#   # }
  
#   # 3. Update each R_i with rejection sampling
#   for (i in 1:n) {
#     yi <- y_data[,,i]
#     repeat {
#       # Compute the concentration matrix C for matrix Fisher distribution
#       C <- (S / sigma2) * t(yi) %*% u + 8 * lambda * t(yi) %*% yi %*% t(yi) %*% u
#       # Sample R_i^T from the matrix Fisher distribution
#       # Note: Ensure that rmf.matrix.gibbs() function is properly defined or use an appropriate sampling method
#       R_i_transpose <- rmf.matrix.gibbs(C, t(R_matrices[,,i]))
#       R_i_proposal <- t(R_i_transpose)

#       # Check if det(R_i) > 0
#       if (det(R_i_proposal) > 0) {
#         R_matrices[,,i] <- R_i_proposal  # Accept the proposal
#         break
#       }
#     }
#   }
  
#   # 4. Update u using manifold optimization
#   # Prepare matrices A, B, and C for the function
  
#   # Initialize A and C_matrix
#   A <- -n * (1 / (2 * sigma2)) * diag(m)  # m x m matrix
#   C_matrix <- matrix(0, nrow = m, ncol = d)  # m x d matrix
  
#   for (i in 1:n) {
#     yi <- y_data[,,i]    # m x d matrix
#     Ri <- R_matrices[,,i] # d x d matrix
    
#     # Update A
#     A <- A - 4 * lambda * yi %*% t(yi)  # m x m matrix
    
#     # Compute t(yi) %*% yi
#     tyi_yi <- t(yi) %*% yi  # d x d matrix
    
#     # Compute yi %*% (tyi_yi)
#     yi_tyi_yi <- yi %*% tyi_yi  # m x d matrix
    
#     # Compute term: yi_tyi_yi %*% t(Ri)
#     term <- yi_tyi_yi %*% t(Ri)  # m x d matrix
    
#     # Update C_matrix
#     C_matrix <- C_matrix + (S / sigma2) * yi %*% t(Ri) + 8 * lambda * term  # m x d matrix
#   }
  
#   # B is an identity matrix of size d x d
#   B <- diag(d)
  
#   # 4. Update u using manifold optimization
#   # Prepare matrices A and C for the optimization
#   A_opt <- A  # m x m symmetric matrix
#   C_opt <- C_matrix  # m x d matrix
  
#   # Initialize u (current estimate)
#   u_init <- u
  
#   # Optimize u
#   u <- optimize_u(u_init, A_opt, C_opt)
  
#   # Store samples for diagnostics
#   S_samples[iter] <- S
#   sigma2_samples[iter] <- sigma2
#   u_samples[,,iter] <- u
#   R_samples[,,,iter] <- R_matrices
  
# }

# cat("\nGibbs Sampling Completed!\n")

# save(S_samples, file = "S_samples_lambda_100.RData")
# save(sigma2_samples, file = "sigma2_samples_lambda_100.RData")
# save(u_samples, file = "u_samples_lambda_100.RData")
# save(R_samples, file = "R_samples_lambda_100.RData")

# cat("Variables have been saved with the suffix '_lambda_100'.\n")

# # Output results
# cat("\nFinal S value:", S, "\n")
# cat("Final sigma^2 value:", sigma2, "\n")
# cat("Final u matrix:\n")
# print(u)


load("res_data_application/S_samples_lambda_100.RData")
load("res_data_application/sigma2_samples_lambda_100.RData")
load("res_data_application/u_samples_lambda_100.RData")
load("res_data_application/R_samples_lambda_100.RData")

B = 5
num_iters = length(sigma2_samples)

# write.table(
#     S_samples, "S_samples_lambda_100.txt",
#     row.names = FALSE, col.names = FALSE)

# write.table(sigma2_samples, "sigma2_samples_lambda_100.txt",
#     row.names = FALSE, col.names = FALSE)

# write.table(u_samples, "u_samples_lambda_100.txt",
#     row.names = FALSE, col.names = FALSE)

# write.table(R_samples, "R_samples_lambda_100.txt",
#     row.names = FALSE, col.names = FALSE)


# plot angle distribution of R

angles_R <- c()
for (b in 1:B) {
    r_b_samples <- R_samples[, , b, ]
    r_b_samples <- matrix(
        aperm(r_b_samples, c(3, 1, 2)), nrow = num_iters, ncol = 25
    )

    angles_R <- c(angles_R, unlist(lapply(1:num_iters, function(i) {

        svd_res <- svd(t(u_samples[, , i]) %*% y_data[, , b] * S_samples[i])
        r_b_optimal <- c(svd_res$u %*% t(svd_res$v))

        cos_angle <- (sum(r_b_optimal * r_b_samples[i, ]) /
                sqrt(sum(r_b_optimal ^ 2)) / sqrt(sum(r_b_samples[i, ] ^ 2))
        )
        acos(cos_angle)
    })))
}


ggplot(data = data.frame(Angle = angles_R), aes(Angle)) +
    geom_density(color = 'black') + theme_bw() +
    theme(
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11)) +
    ylab('Density') + xlim(0, 1.78)

ggsave("density_angle_lambda_100.png", width=3, height=1.5, units='in')














# ---------------------------------------------
# Discard Burn-in Samples and Apply Thinning
# ---------------------------------------------

burn_in <- 10000  # Number of burn-in samples to discard
thin_interval <- 20  # Keep every 20th sample
total_iterations <- length(S_samples)  # Total number of iterations

# Indices of samples after burn-in and thinning
thinned_indices <- seq(burn_in + 1, total_iterations, by = thin_interval)
num_thinned_samples <- length(thinned_indices)

# Subset the samples
S_samples_thinned <- S_samples[thinned_indices]
sigma2_samples_thinned <- sigma2_samples[thinned_indices]
u_samples_thinned <- u_samples[,,thinned_indices]
R_samples_thinned <- R_samples[,,,thinned_indices]


####################


library(ggplot2)
library(scales)  # Improve y-axis readability

# Set random seed
set.seed(123)
num_selected <- 1
selected_indices <- sample(1:dim(R_samples)[3], num_selected, replace = FALSE)
selected_indices <- c(1)

# Get R matrix dimensions
d <- dim(R_samples)[1]
n_selected <- length(selected_indices)
R_samples_selected <- array(0, dim = c(d, d, n_selected, 100000))

# Extract selected R matrix samples
for (s in 1:n_selected) {
  idx <- selected_indices[s]
  R_samples_selected[,,s,] <- R_samples[,,idx,]
}

# Generate R Trace Plots only
for (s in 1:n_selected) {
    idx <- selected_indices[s]
    element_samples <- R_samples_selected[1, 1, s, ]  # Extract [1,1] element of R_{idx}
    
    # Create dataframe
    df <- data.frame(Iteration = c(1:100000), Value = element_samples)
    
    # Trace plot
    trace_plot <- ggplot(df, aes(x = Iteration, y = Value)) +
        geom_line(linewidth=0.1) +
        labs(x = "Iteration", y = parse(text = paste0("R[", idx, "]^{11}"))) +
        scale_x_continuous(breaks = c(0, 30000, 60000, 100000),
            labels = function(x) format(x, scientific = FALSE, big.mark = "")
        ) +
        theme_bw() +
        theme(
            plot.margin = margin(3, 12, 3, 3),
            axis.title.x = element_text(size = 13),
            axis.title.y = element_text(size = 13),
            axis.text.x = element_text(size = 10),
            axis.text.y = element_text(size = 10)
        )
    
    # Save trace plot
    ggsave(
        filename = paste0("trace_R_", idx, "_11_lambda_100.png"),
        plot = trace_plot,
        width = 2.5,
        height = 1.5,
        units = "in"
    )
}

# ---------------------------------------------
# Generate u Trace Plot only
# ---------------------------------------------

# Extract u[1,1] element
u_element_samples <- u_samples[1, 1, ]

# Create dataframe
df_u <- data.frame(Iteration = c(1:100000), Value = u_element_samples)

# u Trace plot
trace_plot_u <- ggplot(df_u, aes(x = Iteration, y = Value)) +
    geom_line(linewidth = 0.1) +
    labs(x = "Iteration", y = expression(u^{11})) +
    scale_x_continuous(breaks = c(0, 30000, 60000, 100000),
            labels = function(x) format(x, scientific = FALSE, big.mark = "")
        ) +
    theme_bw() +
        theme(
            plot.margin = margin(3, 12, 3, 3),
            axis.title.x = element_text(size = 13),
            axis.title.y = element_text(size = 13),
            axis.text.x = element_text(size = 10),
            axis.text.y = element_text(size = 10)
        )

# Save u trace plot
ggsave(
    filename = "trace_u_11_lambda_100.png",
    plot = trace_plot_u,
    width = 2.5,
    height = 1.5,
    units = "in"
)

#####################

library(ggplot2)
library(reshape2)
library(dplyr)

# Set output directory
output_dir <- "plots/"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Set maximum lag value
max_lag <- 40

# Process R matrices -----------------------------------------
# Get R matrix dimensions
d <- dim(R_samples_thinned)[1]
n_matrices <- dim(R_samples_thinned)[3]
n_samples <- dim(R_samples_thinned)[4]

# Create dataframe to store ACF values
R_acf_data <- data.frame()

# Calculate ACF for each element
for (i in 1:d) {
  for (j in 1:d) {
    for (k in 1:n_matrices) {
      # Extract MCMC samples for current element
      element_samples <- R_samples_thinned[i, j, k, ]
      
      # Calculate ACF
      acf_values <- acf(element_samples, lag.max = max_lag, plot = FALSE)
      
      # Store ACF values (including lag=0)
      temp_df <- data.frame(
        row = i,
        col = j,
        matrix_idx = k,
        lag = acf_values$lag,
        acf = acf_values$acf
      )
      
      R_acf_data <- rbind(R_acf_data, temp_df)
    }
  }
}

# Identify min/max values per lag and check outliers
R_extreme_outliers <- R_acf_data %>%
    group_by(lag) %>%
    mutate(
        q1 = quantile(acf, 0.25),
        q3 = quantile(acf, 0.75),
        iqr = q3 - q1,
        lower_bound = q1 - 1.5 * iqr,
        upper_bound = q3 + 1.5 * iqr
    ) %>%
    group_by(lag) %>%
    summarize(
        min_acf = min(acf),
        max_acf = max(acf),
        lower_bound = first(lower_bound),
        upper_bound = first(upper_bound),
        is_min_outlier = min_acf < lower_bound,
        is_max_outlier = max_acf > upper_bound
    )

# Create outlier datasets
R_min_outliers <- R_extreme_outliers %>%
    filter(is_min_outlier) %>%
    select(lag, value = min_acf)

R_max_outliers <- R_extreme_outliers %>%
    filter(is_max_outlier) %>%
    select(lag, value = max_acf)

R_extreme_outliers_final <- rbind(R_min_outliers, R_max_outliers)

# Create R boxplot
R_boxplot <- ggplot(R_acf_data, aes(x = factor(lag), y = acf)) +
    geom_boxplot(
        outlier.shape = NA,
        color = "black",
        fill = "white",
        width = 0.8,
        size = 0.8
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.5) +
    geom_point(
        data = R_extreme_outliers_final,
        aes(x = factor(lag), y = value),
        shape = 16,
        size = 2.5,
        color = "black"
    ) +
    labs(x = "Lag", y = "ACF") +
    scale_x_discrete(breaks = function(x) x[seq(1, length(x), by = 10)]) +
    theme_bw() +
    theme(
        plot.margin = unit(c(5, 15, 5, 5), "mm"),
        axis.title.x = element_text(size = 34, margin = margin(t = 10)),
        axis.title.y = element_text(size = 34, margin = margin(r = 10)),
        axis.text.x = element_text(size = 32),
        axis.text.y = element_text(size = 32),
        axis.ticks = element_line(color = "black"),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA)
    )

# Save R boxplot
ggsave(
    filename = paste0(output_dir, "acf_boxplot_R.png"),
    plot = R_boxplot,
    width = 10,
    height = 5,
    dpi = 300
)

# Process u matrix -----------------------------------------
# Get u matrix dimensions
d_u <- dim(u_samples_thinned)[1]
d_u_cols <- dim(u_samples_thinned)[2]
n_samples_u <- dim(u_samples_thinned)[3]

# Create dataframe to store ACF values
u_acf_data <- data.frame()

# Calculate ACF for each element
for (i in 1:d_u) {
    for (j in 1:d_u_cols) {
        element_samples <- u_samples_thinned[i, j, ]
        acf_values <- acf(element_samples, lag.max = max_lag, plot = FALSE)

        temp_df <- data.frame(
        row = i,
        col = j,
        lag = acf_values$lag,
        acf = acf_values$acf
        )

        u_acf_data <- rbind(u_acf_data, temp_df)
    }
}

# Identify outliers for u matrix
u_extreme_outliers <- u_acf_data %>%
    group_by(lag) %>%
    mutate(
        q1 = quantile(acf, 0.25),
        q3 = quantile(acf, 0.75),
        iqr = q3 - q1,
        lower_bound = q1 - 1.5 * iqr,
        upper_bound = q3 + 1.5 * iqr
    ) %>%
    group_by(lag) %>%
    summarize(
        min_acf = min(acf),
        max_acf = max(acf),
        lower_bound = first(lower_bound),
        upper_bound = first(upper_bound),
        is_min_outlier = min_acf < lower_bound,
        is_max_outlier = max_acf > upper_bound
    )

u_min_outliers <- u_extreme_outliers %>%
  filter(is_min_outlier) %>%
  select(lag, value = min_acf)

u_max_outliers <- u_extreme_outliers %>%
  filter(is_max_outlier) %>%
  select(lag, value = max_acf)

u_extreme_outliers_final <- rbind(u_min_outliers, u_max_outliers)

# Create u boxplot
u_boxplot <- ggplot(u_acf_data, aes(x = factor(lag), y = acf)) +
    geom_boxplot(
        outlier.shape = NA,
        color = "black",
        fill = "white",
        width = 0.8,
        size = 0.8
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 0.5) +
    geom_point(
        data = u_extreme_outliers_final,
        aes(x = factor(lag), y = value),
        shape = 16,
        size = 2.5,
        color = "black"
    ) +
    labs(x = "Lag", y = "ACF") +
    scale_x_discrete(breaks = function(x) x[seq(1, length(x), by = 10)]) +
    theme_bw() +
    theme(
        plot.margin = unit(c(5, 15, 5, 5), "mm"),
        axis.title.x = element_text(size = 34, margin = margin(t = 10)),
        axis.title.y = element_text(size = 34, margin = margin(r = 10)),
        axis.text.x = element_text(size = 32),
        axis.text.y = element_text(size = 32),
        axis.ticks = element_line(color = "black"),
        panel.grid.major = element_line(color = "grey90"),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA)
    )

# Save u boxplot
ggsave(
    filename = paste0(output_dir, "acf_boxplot_u.png"),
    plot = u_boxplot,
    width = 10,
    height = 5,
    dpi = 300
)





####################




library(ggplot2)
library(cluster)
library(clusterSim)
library(psych)
library(aricode)  # For calculating NMI and ARI

num_last_R <- 4500  # Last 100 posterior samples
last_indices <- (4500 - num_last_R + 1):4500  # Sampling indices

# Step 1: Initialize result containers
nmi_values <- numeric(num_last_R)  # NMI for each correction
ari_values <- numeric(num_last_R)  # ARI for each correction
db_indices <- numeric(num_last_R)  # Batch D-B Index for each correction

# Step 2: Process each R_i and calculate metrics
for (j in seq_along(last_indices)) {

    if (j %% 20 == 0) {
        cat("Processing R_i index:", j, "\n")
    }

    Ri_index <- last_indices[j]  # Current R_i index

    # 2.1: Correct batch data
    corrected_data <- list()
    batch_labels <- c()
    cell_labels <- c()

    for (i in 1:n_samples) {
        Xi <- y_data[,,i]  # Current batch data
        Ri <- R_samples_thinned[,,i,Ri_index]  # Posterior sample Ri
        s  <- S_samples_thinned[i]

        # Data correction
        corrected_Xi <- s * Xi %*% t(Ri)
        corrected_data[[i]] <- as.data.frame(corrected_Xi)

        # Store labels
        corrected_data[[i]]$Batch <- rep(i, n_points)
        corrected_data[[i]]$Label <- labels_list[[i]]

        batch_labels <- c(batch_labels, rep(i, n_points))
        cell_labels <- c(cell_labels, labels_list[[i]])
    }

    # Combine all batches
    all_data <- do.call(rbind, corrected_data)

    # 2.2: PCA dimensionality reduction
    pca_result <- prcomp(all_data[, 1:5], scale = TRUE)
    all_data$PC1 <- pca_result$x[, 1]
    all_data$PC2 <- pca_result$x[, 2]

    # 2.3: Clustering analysis
    n_clusters <- n_samples  # Number of clusters equals batch count
    batch_clustering <- pam(all_data[, 1:5], k = n_clusters)
    cluster_labels <- batch_clustering$clustering

    # 2.4: Calculate NMI and ARI
    nmi_values[j] <- tryCatch(
        { NMI(cell_labels, cluster_labels) },
        error = function(e) { return(NA) }
    )

    ari_values[j] <- tryCatch(
        { ARI(cell_labels, cluster_labels) },
        error = function(e) { return(NA) }
    )

    # 2.5: Calculate Batch D-B Index
    db_indices[j] <- tryCatch(
        { index.DB(all_data[, 1:5], cluster_labels, d = NULL, centrotypes = "centroids")$DB },
        error = function(e) { return(NA) }
    )
}


write.table(
    nmi_values, "nmi_values_lambda_100.txt",
    row.names = FALSE, col.names = FALSE)
write.table(
    ari_values, "ari_values_lambda_100.txt",
    row.names = FALSE, col.names = FALSE)
write.table(
    db_indices, "db_indices_lambda_100.txt",
    row.names = FALSE, col.names = FALSE)




# Step 3: Statistical analysis
mean_nmi <- mean(nmi_values, na.rm = TRUE)
mean_ari <- mean(ari_values, na.rm = TRUE)
mean_db_index <- mean(db_indices, na.rm = TRUE)

max_nmi <- max(nmi_values, na.rm = TRUE)
max_nmi_index <- which.max(nmi_values)

max_ari <- max(ari_values, na.rm = TRUE)
max_ari_index <- which.max(ari_values)

min_db_index <- min(db_indices, na.rm = TRUE)
min_db_index_index <- which.min(db_indices)


# Output results
cat("Posterior Mean NMI:", mean_nmi, "\n")
cat("Posterior Mean ARI:", mean_ari, "\n")
cat("Posterior Mean DB Index:", mean_db_index, "\n")
cat("Maximum NMI:", max_nmi, "at index:", max_nmi_index, "\n")
cat("Maximum ARI:", max_ari, "at index:", max_ari_index, "\n")
cat("Minimum DB Index:", min_db_index, "at index:", min_db_index_index, "\n")

# Step 4: Generate histograms
library(ggplot2)


# NMI Histogram
ggplot(data.frame(NMI = nmi_values), aes(x = NMI)) +
    geom_histogram(bins = 12) +
    geom_vline(xintercept = c(0.2984447, 0.3033984), linewidth = 1.1,
                color = c("#f16d6d", "#00a673"), linetype = "dashed") +
    theme_bw() +
    labs(x = "Normalized Mutual Information", y = "Frequency") +
    theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
    )
ggsave("NMI_Histogram_100.png", width=3.5, height=2.2, units="in")

# ARI Histogram
ggplot(data.frame(ARI = ari_values), aes(x = ARI)) +
    geom_histogram(bins = 12) +
    geom_vline(xintercept = c(0.2038716, 0.2638984), linewidth = 1.1,
                color = c("#f16d6d", "#00a673"), linetype = "dashed") +
    theme_bw() +
    labs(x = "Adjusted Rand Index", y = "Frequency") +
    theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
    )
ggsave("ARI_Histogram_100.png", width=3.5, height=2.2, units="in")



# DB Index Histogram
ggplot(data.frame(DBIndex = db_indices), aes(x = DBIndex)) +
    geom_histogram(bins = 12) +
        geom_vline(xintercept = c(1.728992, 1.275662), linewidth = 1.1,
                    color = c("#f16d6d", "#00a673"), linetype = "dashed") +
        theme_bw() +
        labs(x = "Batch Davies\u2013Bouldin index", y = "Frequency") +
        theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
        )

ggsave("DB_Index_Histogram_100.png", width=3.5, height=2.2, units="in")





### Visualization
library(ggplot2)
library(dplyr)

selected_index <- max_ari_index

# Data correction for visualization
corrected_data <- list()
batch_labels <- c()
cell_labels <- c()

Ri_index <- last_indices[selected_index]  
for (i in 1:n_samples) {
    Xi <- y_data[,,i]  # Current batch data
    Ri <- R_samples_thinned[,,i,Ri_index]  
    s  <- S_samples_thinned[Ri_index]

    corrected_Xi <- s * Xi %*% t(Ri)
    corrected_data[[i]] <- as.data.frame(corrected_Xi)

    corrected_data[[i]]$Batch <- rep(i, n_points)
    corrected_data[[i]]$Label <- labels_list[[i]]

    batch_labels <- c(batch_labels, rep(i, n_points))
    cell_labels <- c(cell_labels, labels_list[[i]])
}

# Combine data
all_data <- do.call(rbind, corrected_data)

# PCA transformation
pca_result <- prcomp(all_data[, 1:5], scale = TRUE)
all_data$PC1 <- pca_result$x[, 1]
all_data$PC2 <- pca_result$x[, 2]


# Step 1: Data filtering
corrected_pca_data <- all_data %>%
  filter(PC1 > -50) %>%
  mutate(Batch = as.factor(Batch))


# Step 2: Color mapping
unique_labels <- unique(corrected_pca_data$Label)
unique_batches <- unique(corrected_pca_data$Batch)

label_colors <- setNames(rainbow(length(unique_labels)), unique_labels)
batch_colors <- setNames(rainbow(length(unique_batches)), unique_batches)


mycolors <- c("#f16d6d", "#00a673", "#e69e29", "#00aeea", "#97979b")
batch_colors <- setNames(mycolors, unique_batches)



# Step 3: Plot generation function
generate_corrected_plot <- function(data, aes_col, color_mapping, output_file) {
    plot <- ggplot(data, aes(x = PC1, y = PC2, color = !!aes_col)) +
        geom_point(alpha = 0.9, size = 0.8) +
        scale_color_manual(values = color_mapping) +
        theme_bw() +
        theme(
            legend.position = "none",
            axis.title = element_text(size = 14),
            axis.text = element_text(size = 13)
        ) +
        labs(x = "Principal component 1", y = "Principal component 2")

    ggsave(output_file, plot, width = 2.5, height = 2.5, units = "in")
}


# Step 4: Generate visualizations
generate_corrected_plot(
  data = corrected_pca_data,
  aes_col = sym("Label"),
  color_mapping = label_colors,
  output_file = "PCA_celltype_lambda=100.png"
)


generate_corrected_plot(
  data = corrected_pca_data,
  aes_col = sym("Batch"),
  color_mapping = batch_colors,
  output_file = "PCA_Batch_lambda=100.png"
)
