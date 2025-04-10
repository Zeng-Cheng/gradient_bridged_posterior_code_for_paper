# Load required libraries
library(readr)
library(dplyr)
library(ggplot2)
library(cluster)       # For PAM clustering
library(clusterSim)    # For calculating D-B Index
library(aricode)       # For calculating NMI and ARI
library(vegan)         # For Procrustes Analysis
library(shapes)        # For direct GPA implementation

# Step 1: Data loading
adjusted_files <- list.files(path = "data", pattern = "^sampled_.*\\.csv",
    full.names = TRUE
)

new_matrices <- lapply(adjusted_files, function(file) {
    df <- read_csv(file, show_col_types = FALSE)
    list(data = as.matrix(df[, -ncol(df)]), labels = df[[ncol(df)]])
})


# Step 2: Generalized Procrustes Analysis (GPA)
# Convert data to 3D array format for shapes::procGPA
array_data <- array(
    unlist(lapply(new_matrices, `[[`, "data")),
    dim = c(nrow(new_matrices[[1]]$data),
        ncol(new_matrices[[1]]$data), length(new_matrices)
    )
)

# dim(array_data) = 579 * 5 * 5
for (i in 1:dim(array_data)[3]) {
    array_data[, , i] <- apply(array_data[, , i], 2, scale)
}

# Perform GPA using shapes package
gpa_result <- procGPA(array_data)
aligned_data <- gpa_result$rotated  # Aligned data (3D array)

# Convert aligned data to 2D matrix and extract labels
aligned_data_2d <- do.call(
    rbind, lapply(1:dim(aligned_data)[3], function(i) aligned_data[, , i])
)

aligned_labels <- unlist(lapply(new_matrices, `[[`, "labels"))
aligned_batches <- unlist(lapply(seq_along(new_matrices), function(i) {
    rep(i, nrow(new_matrices[[i]]$data))  # Use numeric batch labels
}))

# Calculate NMI and ARI after clustering
# PAM clustering using first 5 principal components
# pca_result <- prcomp(aligned_data_2d, center = TRUE, scale. = TRUE)
# n_pcs <- 5  # Use first 5 PCs
# pca_data <- pca_result$x[, 1:n_pcs]

# Cluster number equals batch number
n_clusters <- length(unique(aligned_batches))
batch_clustering <- pam(aligned_data_2d, k = n_clusters)

# Calculate NMI and ARI (using clustering results vs true batch labels)
nmi_score <- NMI(batch_clustering$clustering, aligned_labels)
ari_score <- ARI(batch_clustering$clustering, aligned_labels)
cat("GPA-aligned Batch NMI: ", nmi_score, "\n")
cat("GPA-aligned Batch ARI: ", ari_score, "\n")

write.table(aligned_data_2d, "gpa_aligned_data_2d.txt", row.names = FALSE, col.names = FALSE)


# Calculate Batch D-B Index
db_index <- index.DB(
    aligned_data_2d, batch_clustering$clustering, d = NULL, centrotypes = "centroids"
)$DB
cat("GPA-aligned Batch D-B Index: ", db_index, "\n")


#############################################

# Visualization
pca_plot_data <- data.frame(
    PC1 = pca_data[, 1],
    PC2 = pca_data[, 2],
    CellType = aligned_labels,
    Batch = aligned_batches
)

##-------


# Step 1: Data filtering
pca_plot_data_filtered <- pca_plot_data %>%
    filter(PC1 > -50, PC2 > -3, PC1 < 60, PC2 < 5)
    # Keep only data with PC1 > -50

# Ensure Batch is factor
pca_plot_data_filtered <- pca_plot_data_filtered %>%
    mutate(Batch = as.factor(Batch))  # Convert Batch to factor

# Step 2: Ensure correct color mapping for CellType and Batch
# Get unique values
unique_celltypes <- unique(pca_plot_data_filtered$CellType)
unique_batches <- unique(pca_plot_data_filtered$Batch)

# Create color mappings
celltype_colors <- setNames(rainbow(length(unique_celltypes)), unique_celltypes)
batch_colors <- setNames(rainbow(length(unique_batches)), unique_batches)

mycolors <- c("#f16d6d", "#00a673", "#e69e29", "#00aeea", "#97979b")
batch_colors <- setNames(mycolors, unique_batches)



# Step 3: Main plot generation function
generate_pca_plot <- function(data, aes_col, color_mapping, output_file) {
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

# Step 4: Generate PCA plots
generate_pca_plot(
    data = pca_plot_data_filtered,
    aes_col = sym("CellType"),
    color_mapping = celltype_colors,
    output_file = "vis/PCA_CellType_gpa.png"
)

generate_pca_plot(
    data = pca_plot_data_filtered,
    aes_col = sym("Batch"),
    color_mapping = batch_colors,
    output_file = "PCA_Batch_gpa.png"
)
