library(ggplot2)

vec1 <- c(unlist(read.table("res_data_application/angles_gibbs.txt")))
vec2 <- c(unlist(read.table("res_data_application/angles_gradient-bridge.txt")))


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
    geom_histogram(bins = 15) + xlim(c(4, 23)) +
    geom_vline(xintercept = c(9.7343), linewidth = 1.1,
                color = c("#f16d6d"), linetype = "dashed") +
    theme_bw() +
    labs(x = "Batch Davies\u2013Bouldin index", y = "Frequency") +
    theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
    )
ggsave("DB_Index_Histogram_gradient-bridge.png", width=3.5, height=2.2, units="in")



db_indices = c(unlist(read.table("res_data_application/db_indices_gibbs.txt")))

# DB Index Histogram
ggplot(data.frame(DBIndex = db_indices), aes(x = DBIndex)) +
    geom_histogram(bins = 15) + xlim(c(4, 23)) +
    geom_vline(xintercept = c(9.7343), linewidth = 1.1,
                color = c("#f16d6d"), linetype = "dashed") +
    theme_bw() +
    labs(x = "Batch Davies\u2013Bouldin index", y = "Frequency") +
    theme(
        axis.title = element_text(size = 11),
        axis.text = element_text(size = 9)
    )
ggsave("DB_Index_Histogram_gibbs.png", width=3.5, height=2.2, units="in")




# Load required libraries
library(MASS)
library(rstiefel)
library(ManifoldOptim)
library(Matrix)
library(progress)
library(msm)
library(readr)


# Read adjusted matrices
adjusted_files <- list.files(path = "data", pattern = "^sampled_.*\\.csv",
    full.names = TRUE
)

new_matrices <- lapply(adjusted_files, function(file) {
    df <- read_csv(file, show_col_types = FALSE)
    list(data = as.matrix(df[, -ncol(df)]), labels = df[[ncol(df)]])
    # Separate data and labels
})

n_points <- 579  # Number of samples (n_min)
n_batches <- length(new_matrices)  # Number of batches (files)

labels_list <- vector("list", n_batches)  # Store labels
for (i in seq_along(new_matrices)) {
    # Store labels
    labels_list[[i]] <- new_matrices[[i]]$labels
}

cat("Labels for each group are stored in labels_list.\n")



########################################
######### Visualization #########
########################################

library(ggplot2)
library(dplyr)

batch_labels <- c()
cell_labels <- c()

for (i in 1:n_batches) {
    batch_labels <- c(batch_labels, rep(i, n_points))
    cell_labels <- c(cell_labels, labels_list[[i]])
}

# Color mapping
unique_labels <- unique(cell_labels)
unique_batches <- unique(batch_labels)

# label_colors <- setNames(rainbow(length(unique_labels)), unique_labels)
# batch_colors <- setNames(rainbow(length(unique_batches)), unique_batches)

mycolors <- c("#f16d6d", "#00a673", "#e69e29", "#00aeea", "#97979b")
batch_colors <- setNames(mycolors, unique_batches)
label_colors <- setNames(mycolors, unique_labels)

# Plot generation function
generate_corrected_plot <- function(data, aes_col, color_mapping, output_file) {
    plot <- ggplot(data, aes(x = PC1, y = PC2, color = !!aes_col)) +
        geom_point(alpha = 0.5, size = 0.2) +
        scale_color_manual(values = color_mapping) +
        theme_bw() +
        theme(
            plot.margin = unit(c(0.2, 0.7, 0, 0), "cm"), 
            # top, right, bottom, left
            legend.position = "none",
            axis.title = element_text(size = 14),
            axis.text = element_text(size = 13)
        ) +
        labs(x = "Principal component 1", y = "Principal component 2")

    ggsave(output_file, plot, width = 2.5, height = 2.5, units = "in")
}

################################

####### gradient-bridged #######

pca_res <- read.table(
    "res_data_application/pca_gradient-bridge.txt") / 100

all_data = data.frame(pca_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    filter(PC1 < 10, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_lambda=100.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_lambda=100.png")


####### Gibbs posterior

pca_res <- read.table(
    "res_data_application/pca_gibbs.txt")


all_data = data.frame(pca_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    filter(PC1 > -10, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_lambda=0.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_lambda=0.png")

##### raw data

pca_res <- read.table("res_data_application/pca_raw.txt")


all_data = data.frame(pca_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    filter(PC1 > -0.5, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_rawdata.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_rawdata.png")


# GPA

pca_res <- read.table("res_data_application/pca_gpa.txt")


all_data = data.frame(pca_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    # filter(PC1 < 50, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_gpa.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_gpa.png")



#############################
### UMAP ###

####### gradient-bridged #######

umap_res <- read.table(
    "res_data_application/umap_gradient-bridge.txt")

all_data = data.frame(umap_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    # filter(PC1 < 50, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_lambda=100.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_lambda=100.png")


####### Gibbs posterior

umap_res <- read.table(
    "res_data_application/umap_gibbs.txt")


all_data = data.frame(umap_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    # filter(PC1 < 50, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_lambda=0.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_lambda=0.png")

##### raw data

umap_res <- read.table("res_data_application/umap_raw.txt")


all_data = data.frame(umap_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    # filter(PC1 < 50, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_rawdata.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_rawdata.png")


# GPA

umap_res <- read.table("res_data_application/umap_gpa.txt")


all_data = data.frame(umap_res)
colnames(all_data) = c("PC1", "PC2")
all_data$Label = cell_labels
all_data$Batch = batch_labels

# Data filtering
corrected_pca_data <- all_data %>%
    # filter(PC1 < 50, PC2 < 50) %>%
    mutate(Batch = as.factor(Batch))

# Generate visualizations
generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Label"),
    color_mapping = label_colors, output_file = "PCA_celltype_gpa.png")

generate_corrected_plot(data = corrected_pca_data, aes_col = sym("Batch"),
    color_mapping = batch_colors, output_file = "PCA_Batch_gpa.png")