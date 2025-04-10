library(ggplot2)

vec1 <- c(unlist(read.table("angles_gibbs.txt")))
vec2 <- c(unlist(read.table("angles_gradient-bridge.txt")))


# Plot density
ggplot(data.frame(value=vec1), aes(x = value)) +
    geom_density() +
    labs(x = "Angle", y = "Density") +
    theme_bw() +
    theme(
        axis.title = element_text(size = 8),
        axis.text = element_text(size = 6)
    )

ggsave("angels_gibbs.png", width=2.2, height=1.2, units="in")

ggplot(data.frame(value=vec2), aes(x = value)) +
    geom_density() +
    labs(x = "Angle", y = "Density") +
    theme_bw() +
    xlim(0, 0.2) +
    theme(
        axis.title = element_text(size = 8),
        axis.text = element_text(size = 6)
    )

ggsave("angels_gradient-bridge.png", width=2.2, height=1.2, units="in")




db_indices = c(unlist(read.table("res_data_application/db_indices_gradient-bridge.txt")))

# DB Index Histogram
ggplot(data.frame(DBIndex = db_indices), aes(x = DBIndex)) +
    geom_histogram(bins = 15) +
    geom_vline(xintercept = c(1.032, 0.803), linewidth = 1.1,
                color = c("#f16d6d", "#00a673"), linetype = "dashed") +
    theme_bw() +
    labs(x = "Batch Davies\u2013Bouldin index", y = "Frequency") +
    theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
    )
ggsave("DB_Index_Histogram_gradient-bridge.png", width=3.5, height=2.2, units="in")







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
        stop(paste("The dimensions of adjusted matrix", adjusted_files[i], "are not 638 x 5."))
    }

    # Row-wise scaling of the matrix
    scaled_matrix <- apply(current_matrix, 2, scale)  # Standardize each row

    # Populate y_data
    y_data[,,i] <- scaled_matrix

    # Store labels
    labels_list[[i]] <- current_labels
}

# Step 5: Output results
cat("The dimensions of y_data are:", dim(y_data), "\n")
cat("Labels for each group are stored in labels_list.\n")


### Visualization

library(ggplot2)
library(dplyr)


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



pca_python <- read.table("pca_gradient-bridge.txt")
all_data$PC1 <- pca_python[, 1] / 100000
all_data$PC2 <- pca_python[, 2] / 100000



# Step 1: Data filtering
corrected_pca_data <- all_data %>%
    filter(PC1 < 5, PC2 < 5) %>%
    mutate(Batch = as.factor(Batch))


# Step 2: Color mapping
unique_labels <- unique(corrected_pca_data$Label)
unique_batches <- unique(corrected_pca_data$Batch)

label_colors <- setNames(rainbow(length(unique_labels)), unique_labels)
batch_colors <- setNames(rainbow(length(unique_batches)), unique_batches)


mycolors <- c("#f16d6d", "#00a673", "#e69e29", "#00aeea", "#97979b")
batch_colors <- setNames(mycolors, unique_batches)
label_colors <- setNames(mycolors, unique_labels)


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
