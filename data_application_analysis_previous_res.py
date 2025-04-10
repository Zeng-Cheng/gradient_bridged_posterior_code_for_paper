# %%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import davies_bouldin_score as DBI

from tqdm.notebook import trange
import seaborn as sns
from numpy.linalg import svd

# %%
# ====================================
# Load Data
# ====================================

data_path = "data"
adjusted_files = glob.glob(os.path.join(data_path, "sampled_*.csv"))

# Initialize lists for data
X_data_list = []
labels_list = []

# Read CSV files and process matrices
for file in adjusted_files:
    df = pd.read_csv(file)
    X_matrix = jnp.array(df.iloc[:, :-1].values, dtype=jnp.float32)
    labels = df.iloc[:, -1].values  # Extract labels

    # Standardize rows
    X_matrix = (X_matrix - X_matrix.mean(axis=0, keepdims=True)) / (X_matrix.std(axis=0, keepdims=True) + 1e-6)
    # X_matrix = X_matrix / 100

    X_data_list.append(X_matrix.T)
    labels_list.append(labels)

X_data = jnp.stack(X_data_list, axis=0)  # Shape: (B, d, n)
B, d, n = X_data.shape
print(f"Data Loaded: B={B}, n={n}, d={d}")


# %%

gpa_res = np.loadtxt("gpa_aligned_data_2d.txt")

# Compute corrections and clustering analysis
corrected_data = gpa_res
batch_labels, cell_labels = [], []

for b in range(B):
    batch_labels.extend([b] * n)
    cell_labels.extend(labels_list[b])

# Clustering analysis
kmeans = KMeans(n_clusters=B, n_init=10).fit(corrected_data)
cluster_labels = kmeans.labels_

# Compute NMI and ARI
print(f"NMI: {NMI(cell_labels, cluster_labels)}")
print(f"ARI: {ARI(cell_labels, cluster_labels)}")

# Compute Batch Davies-Bouldin Index
print(f"DBI: {DBI(corrected_data, cluster_labels)}")




# %%

# Load s_samples, u_samples, and sigma2_samples
s_samples = np.loadtxt("S_samples_lambda_0_new.txt")
sigma2_samples = np.loadtxt("sigma2_samples_lambda_0_new.txt")

# Reshape u_samples back to its original form (num_samples, d, n)
num_samples = s_samples.shape[0]  # Assuming s_samples has shape (num_samples,)
u_samples = np.loadtxt("u_samples_lambda_0_new.txt")
u_samples = u_samples.reshape(n, d, num_samples, order='F').transpose(2, 1, 0)

R_samples = np.loadtxt("R_samples_lambda_0_new.txt")
R_samples = R_samples.reshape(d, d, B, num_samples, order='F').transpose(3, 2, 0, 1)

# Parameters
thin_interval = 20  # Keep every 20th sample
num_thinned_samples = 500

# Indices of samples after burn-in and thinning
thinned_indices = np.arange(2000, num_samples, thin_interval)

# Subset the samples
S_samples_thinned = s_samples[thinned_indices]
sigma2_samples_thinned = sigma2_samples[thinned_indices]
u_samples_thinned = u_samples[thinned_indices]
R_samples_thinned = R_samples[thinned_indices]

# Processing posterior samples
nmi_values = np.zeros(num_thinned_samples)
ari_values = np.zeros(num_thinned_samples)
db_indices = np.zeros(num_thinned_samples)

# Compute corrections and clustering analysis
for j in trange(num_thinned_samples):
    
    corrected_data = []
    batch_labels, cell_labels = [], []
    
    for b in range(B):
        Xb = X_data[b, :, :]  # Current batch data
        Ri = R_samples_thinned[j, b, :, :]  # Posterior sample Ri
        # s = S_samples_thinned[j, b]
        s = S_samples_thinned[j]
        
        # Data correction
        corrected_Xb = jnp.matmul(Ri, Xb) * s
        corrected_data.append(corrected_Xb.T)
        
        # Store labels
        batch_labels.extend([b] * n)
        cell_labels.extend(labels_list[b])
    
    all_data = np.vstack(corrected_data)
    # print(f"all_data shape: {all_data.shape}")
    
    # Clustering analysis
    kmeans = KMeans(n_clusters=B, n_init=10).fit(all_data)
    cluster_labels = kmeans.labels_
    
    # Compute NMI and ARI
    nmi_values[j] = NMI(cell_labels, cluster_labels)
    ari_values[j] = ARI(cell_labels, cluster_labels)
    
    # Compute Batch Davies-Bouldin Index
    db_indices[j] = DBI(all_data, cluster_labels)

# %%
# Compute summary statistics
mean_nmi, mean_ari, mean_db_index = np.nanmean(nmi_values), np.nanmean(ari_values), np.nanmean(db_indices)
max_nmi, max_nmi_index = np.nanmax(nmi_values), np.nanargmax(nmi_values)
max_ari, max_ari_index = np.nanmax(ari_values), np.nanargmax(ari_values)
min_db_index, min_db_index_index = np.nanmin(db_indices), np.nanargmin(db_indices)

print(f"Posterior Mean NMI: {mean_nmi}")
print(f"Posterior Mean ARI: {mean_ari}")
print(f"Posterior Mean DB Index: {mean_db_index}")
print(f"Maximum NMI: {max_nmi} at index: {max_nmi_index}")
print(f"Maximum ARI: {max_ari} at index: {max_ari_index}")
print(f"Minimum DB Index: {min_db_index} at index: {min_db_index_index}")


# %%
# Plot histogram
plt.figure(figsize=(5, 3))
plt.hist(db_indices, bins=12, color="#00a673", edgecolor="black", alpha=0.7)
plt.xlabel("Batch Davies–Bouldin Index")
plt.ylabel("Frequency")
plt.title("Histogram of DB Index")
plt.grid(True, linestyle="--", alpha=0.6)

# Save or show plot
plt.show()



# %%
angles_R = []

for b in range(B):
    r_b = R_samples_thinned[:, b, :, :].reshape(num_thinned_samples, -1)

    X = X_data[b, :, :] # shape: (d, n)
    s_b = S_samples_thinned
    u = u_samples_thinned # shape: (num_samples, d, n)

    # Compute svd of s_b * u @ X.T
    s_bu_XT = s_b[:, None, None] * u @ X.T  # Shape: (num_samples, d, d)
    u_svd, _, v_svd = svd(s_bu_XT)
    r_b_optimal = (u_svd @ v_svd).reshape(num_thinned_samples, -1)

    cos_angle = np.sum(r_b_optimal * r_b, axis=1) / (
        np.linalg.norm(r_b_optimal, axis=1) * np.linalg.norm(r_b, axis=1)
    )
    angles_R.append(np.arccos(cos_angle))

# Plot density using seaborn
plt.figure(figsize=(3, 1.5))
sns.kdeplot(angles_R, color='black')
plt.xlabel("Angle", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(False)
plt.tight_layout()

# Save the plot
# plt.savefig("density_angle_lambda_100.png", dpi=300)
plt.show()
# %%
