library(Seurat)
library(SeuratData)
library(SeuratDisk)
library(ggplot2)
library(dplyr)

# # Install and load the panc8 dataset
# InstallData("panc8")  
# panc8 <- UpdateSeuratObject(panc8)

# # Split the dataset by the "tech" column (technology platform)
# panc8.list <- SplitObject(panc8, split.by = "tech")  

# # Normalize and identify variable features for each subset
# panc8.list <- lapply(panc8.list, function(x) {
#   x <- NormalizeData(x)                             
#   x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)  
#   return(x)
# })

# # Perform PCA on each subset and reduce dimensions to 5
# panc8.list <- lapply(panc8.list, function(x) {
#   # Normalize the data
#   x <- NormalizeData(x)                             
#   # Select highly variable genes
#   x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)  
#   # Scale the data
#   x <- ScaleData(x, features = VariableFeatures(x))
#   # Perform PCA and reduce dimensions to 5
#   x <- RunPCA(x, features = VariableFeatures(x), npcs = 5)
#   return(x)
# })

# # Extract the PCA matrix for each subset
# pca_matrices <- lapply(panc8.list, function(x) {
#   # Extract the PCA embedding matrix (5 components × number of samples)
#   pca_matrix <- Embeddings(x, reduction = "pca")[, 1:5]  # Extract the first 5 principal components
#   return(pca_matrix)
# })

# # Add cell type labels to the PCA matrices
# pca_matrices_with_labels <- lapply(names(panc8.list), function(name) {
#   # Get the PCA matrix for the current subset
#   pca_matrix <- Embeddings(panc8.list[[name]], reduction = "pca")[, 1:5]
#   # Get the cell type labels from the metadata
#   cell_types <- panc8.list[[name]]@meta.data$celltype
#   # Combine PCA matrix with cell type labels
#   pca_with_labels <- data.frame(pca_matrix, CellType = cell_types)
#   return(pca_with_labels)
# })

# # Save the PCA matrices with cell type labels to files
# for (i in seq_along(pca_matrices_with_labels)) {
#   write.csv(pca_matrices_with_labels[[i]], 
#             file = paste0("pca_matrix_", names(panc8.list)[i], ".csv"), 
#             row.names = TRUE)
# }


# # Load required libraries
# library(readr)
# library(dplyr)

# # Step 1: Read stored files
# files <- list.files(pattern = "^pca_matrix_.*\\.csv")
# data_list <- lapply(files, read_csv, show_col_types = FALSE)
# names(data_list) <- gsub("\\.csv", "", files)

# # Step 2: Extract PCA matrices and labels
# pca_matrices <- lapply(data_list, function(df) {
#   mat <- as.matrix(df[, -c(1, ncol(df))])  # Extract PCA matrix (remove first and last columns)
#   mat <- t(mat)  # Transpose matrix
#   return(mat)
# })
# cell_labels <- lapply(data_list, function(df) df[[ncol(df)]])   # Extract labels

# # Step 3: Identify and process the batch with smallest sample size
# batch_sizes <- sapply(pca_matrices, ncol)
# min_batch_size <- min(batch_sizes)
# min_batch_index <- which.min(batch_sizes)
# min_batch_name <- names(data_list)[min_batch_index]

# cat("Batch with the smallest sample size:", min_batch_name, "\n")
# cat("Minimum batch size before filtering:", min_batch_size, "\n")

# # Extract matrix and labels from smallest batch
# min_batch_matrix <- pca_matrices[[min_batch_index]]
# min_batch_labels <- cell_labels[[min_batch_index]]

# # Remove cell types with ≤20 samples
# label_counts <- table(min_batch_labels)
# valid_labels <- names(label_counts[label_counts > 20])
# filtered_indices <- which(min_batch_labels %in% valid_labels)
# filtered_matrix <- min_batch_matrix[, filtered_indices, drop = FALSE]
# filtered_labels <- min_batch_labels[filtered_indices]

# cat("Minimum batch size after filtering:", ncol(filtered_matrix), "\n")
# cat("Remaining cell types in the minimum batch:\n")
# print(table(filtered_labels))

# # Step 4: Stratified random sampling for other batches
# final_sample_size <- ncol(filtered_matrix)  # Minimum sample size
# final_cell_types <- unique(filtered_labels)  # Cell types in minimum batch

# sampled_matrices <- list()
# sampled_labels <- list()

# for (i in seq_along(pca_matrices)) {
#     Xi <- pca_matrices[[i]]
#     current_labels <- cell_labels[[i]]
    
#     # Keep only matching cell types
#     valid_indices <- which(current_labels %in% final_cell_types)
#     Xi <- Xi[, valid_indices, drop = FALSE]
#     current_labels <- current_labels[valid_indices]
    
#     # Stratified sampling by cell type
#     sampled_indices <- unlist(lapply(final_cell_types, function(label) {
#         label_indices <- which(current_labels == label)
#         if (length(label_indices) == 0) {
#         warning(paste("Skipping cell type:", label, "in batch:", names(data_list)[i], "- No samples found."))
#         return(NULL)
#         }
#         if (length(label_indices) < floor(final_sample_size / length(final_cell_types))) {
#         # Keep all samples for cell types with insufficient count
#         return(label_indices)
#         } else {
#         # Proportional sampling for sufficient cell types
#         return(sample(label_indices, floor(final_sample_size / length(final_cell_types))))
#         }
#     }))
    
#     # Supplement with random samples if needed
#     if (length(sampled_indices) < final_sample_size) {
#         remaining_indices <- setdiff(seq_len(ncol(Xi)), sampled_indices)
#         extra_samples <- sample(remaining_indices, final_sample_size - length(sampled_indices))
#         sampled_indices <- c(sampled_indices, extra_samples)
#     }
    
#     # Validate final sample size
#     stopifnot(length(sampled_indices) == final_sample_size)
    
#     # Save results
#     sampled_matrices[[names(data_list)[i]]] <- Xi[, sampled_indices, drop = FALSE]
#     sampled_labels[[names(data_list)[i]]] <- current_labels[sampled_indices]
    
#     cat("Batch:", names(data_list)[i], "- Final sampled size:", length(sampled_indices), "\n")
# }

# # Step 5: Export sampling results
# for (name in names(sampled_matrices)) {
#   final_data <- data.frame(t(sampled_matrices[[name]]))  # Restore original format
#   final_data$CellType <- sampled_labels[[name]]
#   write.csv(final_data, paste0("sampled_", name, ".csv"), row.names = FALSE)
# }

# cat("Sampling completed. Results saved to 'sampled_' files.\n")

# Analysis and Visualization Section ----
library(readr)
library(dplyr)
library(ggplot2)
library(cluster)       # For PAM clustering
library(clusterSim)    # For D-B Index calculation
library(aricode)       # For NMI/ARI calculation

# Step 1: Read sampled files
sampled_files <- list.files(path = "data", pattern = "^sampled_.*\\.csv",
    full.names = TRUE
)

sampled_data_list <- lapply(sampled_files, read_csv, show_col_types = FALSE)

# Extract batch info from filenames
file_batches <- gsub("^sampled_|\\.csv$", "", sampled_files)

# Step 2: Merge all sampled data
all_pca_data <- NULL
all_cell_labels <- NULL
all_tech_labels <- NULL

for (i in seq_along(sampled_data_list)) {
    df <- sampled_data_list[[i]]

    # Extract PCA data
    pca_matrix <- as.matrix(df[, -ncol(df)])
    pca_matrix <- apply(pca_matrix, 2, scale)
    all_pca_data <- rbind(all_pca_data, pca_matrix)

    # Extract labels
    all_cell_labels <- c(all_cell_labels, df$CellType)
    all_tech_labels <- c(all_tech_labels, rep(file_batches[i], nrow(df)))
}

cat("Total samples after merging all sampled data:", nrow(all_pca_data), "\n")

# Step 3: Calculate Batch D-B Index
n_clusters <- length(unique(all_tech_labels))
batch_clustering <- pam(all_pca_data[, 1:5], k = n_clusters)
db_index <- index.DB(all_pca_data[, 1:5], batch_clustering$clustering,
    d = NULL, centrotypes = "centroids"
)$DB
cat("Batch D-B Index: ", db_index, "\n")

# Step 4: Calculate NMI and ARI
nmi_value <- NMI(batch_clustering$clustering, all_cell_labels)
ari_value <- ARI(batch_clustering$clustering, all_cell_labels)

cat("NMI between Clustering and Cell Labels:\n")
print(nmi_value)

cat("ARI between Clustering and Cell Labels:\n")
print(ari_value)



######################


# Step 5: Data visualization
# Load required packages
library(ggplot2)
library(dplyr)

pca_data <- data.frame(
  all_pca_data,
  CellType = all_cell_labels,
  Tech = all_tech_labels
)
colnames(pca_data)[1:2] <- c("PC1", "PC2")


# Step 1: Data filtering
pca_data_filtered <- pca_data %>%
    filter(PC1 > -6, PC2 > -5) %>%  # Keep only data with PC1 > -50
    mutate(Tech = as.factor(as.numeric(as.factor(Tech))))  # Convert Tech to numeric factor

# Step 2: Ensure correct color mapping for CellType and Batch
# Get unique values
unique_celltypes <- unique(pca_data_filtered$CellType)
unique_batches <- unique(pca_data_filtered$Tech)

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



# Step 5: Generate main PCA plots
generate_pca_plot(
  data = pca_data_filtered,
  aes_col = sym("CellType"),
  color_mapping = celltype_colors,
  output_file = "vis/PCA_CellType_rawdata.png"
)


generate_pca_plot(
  data = pca_data_filtered,
  aes_col = sym("Tech"),
  color_mapping = batch_colors,
  output_file = "PCA_Batch_rawdata.png"
)



# Step 4: Legend generation function
library(ggplot2)
library(cowplot)

generate_legend <- function(unique_values, color_mapping, output_file) {
    # Create dummy dataframe with Label column
    legend_data <- data.frame(Label = unique_values)

    # Create empty plot for legend generation only
    dummy_plot <- ggplot(legend_data, aes(x = 1, y = 1, color = Label)) +
        geom_point(alpha = 0) +  # Use transparent points
        scale_color_manual(values = color_mapping) +
        theme_bw() +
        theme(
            legend.position = "bottom",
            legend.title = element_blank(),
            legend.text = element_text(size = 9)
        ) +
        guides(color = guide_legend(override.aes = list(alpha = 1, size = 3)))

    # Save legend as separate image
    ggsave(output_file, dummy_plot, width = 3, height = 1, units = "in")
}


# Step 6: Generate standalone legends
generate_legend(
  unique_values = unique_celltypes,
  color_mapping = celltype_colors,
  output_file = "vis/Legend_CellType.png"
)


generate_legend(
    unique_values = unique_batches,
    color_mapping = batch_colors,
    output_file = "Legend_Batch.png"
)
