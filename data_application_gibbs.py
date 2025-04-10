# %%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pandas as pd
import glob
import os

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import seaborn as sns
from numpy.linalg import svd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import davies_bouldin_score as DBI

from tqdm.notebook import trange

# %%
# Set seeds for reproducibility
np.random.seed(4812)
# JAX uses PRNG keys for randomness; here we set an initial key.
key = jax.random.PRNGKey(123)

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

@jax.jit
def batch_log_likelihood(R, X, s, u, sigma2, lambda_reg=100.0):
    logL = 0.0
    for b in range(B):
        R_b = R[b, :, :]
        norm_term = jnp.linalg.norm(R_b @ X[b] - s[b] * u) ** 2
        reg = jnp.sum((R_b.T @ R_b - jnp.eye(d)) ** 2)
        logL += -0.5 * B * jnp.log(sigma2) 
        logL += - norm_term / (2 * sigma2) - lambda_reg * reg
    return logL

def model(X_data, lambda_reg=100.0):
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(2.0, 10.0))
    s = numpyro.sample("s", dist.HalfNormal(1.0), sample_shape=(B,))
    u = numpyro.sample("u", dist.Normal(0., 1.), sample_shape=(d, n))
    R = numpyro.sample("R", dist.Normal(0.0, 10.0), sample_shape=(B, d, d))

    logL = batch_log_likelihood(R, X_data, s, u, sigma2, lambda_reg=lambda_reg)
    numpyro.factor("likelihood", logL)

# %%
# Run MCMC with NUTS
lambda_reg = 500.0
nuts_kernel = NUTS(model)
num_samples = 10000

key, subkey = jax.random.split(key)
mcmc = MCMC(nuts_kernel, num_warmup=5000, num_samples=num_samples, num_chains=1)
mcmc.run(subkey, X_data, lambda_reg=lambda_reg)


# %%
# Extract Results
posterior_samples = mcmc.get_samples()
s_samples = posterior_samples["s"]
u_samples = posterior_samples["u"]
sigma2_samples = posterior_samples["sigma2"]
R_samples = posterior_samples["R"]

# %%

rtest = R_samples[0, 0, :, :]
rtest.T @ rtest

# %%
# Save s_samples, u_samples, and sigma2_samples
np.savetxt("s_samples_gibbs.txt", np.array(s_samples), delimiter=",")
np.savetxt("u_samples_gibbs.txt", np.array(u_samples).reshape(u_samples.shape[0], -1), delimiter=",")  # Flatten last two dims
np.savetxt("sigma2_samples_gibbs.txt", np.array(sigma2_samples), delimiter=",")

R_samples_reshaped = R_samples.reshape(num_samples * B, d * d)
np.savetxt("R_samples_gibbs.txt", R_samples_reshaped, delimiter=",")

# %%
# Load s_samples, u_samples, and sigma2_samples
# s_samples = np.loadtxt("res_data_application/s_samples.txt", delimiter=",")
# sigma2_samples = np.loadtxt("res_data_application/sigma2_samples.txt", delimiter=",")

# # Reshape u_samples back to its original form (num_samples, d, n)
# num_samples = s_samples.shape[0]  # Assuming s_samples has shape (num_samples,)
# u_samples = np.loadtxt("res_data_application/u_samples.txt", delimiter=",").reshape(num_samples, d, n)


# %%
# Extract W_1_11 (first entry of first batch's W matrix)
R_1_11_samples = R_samples[:, 0, 0, 0]  # Extract (1,1) entry

# Extract u_11 (first element of u)
u_11_samples = u_samples[:, 0, 0]

# Function to plot trace and ACF
def plot_trace_acf(samples, title, ax_trace, ax_acf):
    ax_trace.plot(samples, alpha=0.6)
    ax_trace.set_title(f"Trace Plot: {title}")
    ax_trace.set_xlabel("Iteration")
    
    acf_vals = acf(samples, nlags=40, fft=True)
    ax_acf.stem(range(len(acf_vals)), acf_vals)
    ax_acf.set_title(f"ACF: {title}")
    ax_acf.set_xlabel("Lag")

# Plot all variables
fig, axes = plt.subplots(2, 4, figsize=(8, 4))  

# Plot sigma2
plot_trace_acf(sigma2_samples, r"$\sigma^2$", axes[0, 1], axes[1, 1])

# Plot W_1_11
plot_trace_acf(R_1_11_samples, r"$R_{1,11}$", axes[0, 2], axes[1, 2])

# Plot u_11
plot_trace_acf(u_11_samples, r"$u_{11}$", axes[0, 0], axes[1, 0])

# Plot s
plot_trace_acf(s_samples[:, 0], r"$s$", axes[0, 3], axes[1, 3])

plt.tight_layout()
plt.show()




# %%

thin_interval = 10  # Keep every 20th sample
num_thinned_samples = num_samples // thin_interval

# Indices of samples after burn-in and thinning
thinned_indices = np.arange(0, num_samples, thin_interval)

# Subset the samples
s_samples_thinned = s_samples[thinned_indices]
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
        s = s_samples_thinned[j, b]
        
        # Data correction
        corrected_Xb = jnp.matmul(Ri, Xb) / s
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
j = max_nmi_index
corrected_data = []

for b in range(B):
    Xb = X_data[b, :, :]  # Current batch data
    Ri = R_samples_thinned[j, b, :, :]  # Posterior sample Ri
    s = s_samples_thinned[j, b]
    
    # Data correction
    corrected_Xb = jnp.matmul(Ri, Xb) / s
    corrected_data.append(corrected_Xb.T)


all_data = np.vstack(corrected_data)

pca = PCA(n_components=2)
all_data_pca = pca.fit_transform(all_data)

np.savetxt("pca_gibbs.txt", all_data_pca)



# %%
db_indices_np = jnp.asarray(db_indices)

# Plot histogram
plt.figure(figsize=(5, 3))
plt.hist(db_indices_np, bins=12, color="#00a673", edgecolor="black", alpha=0.7)
plt.xlabel("Batch Davies–Bouldin Index")
plt.ylabel("Frequency")
plt.title("Histogram of DB Index")
plt.grid(True, linestyle="--", alpha=0.6)

# Save or show plot
plt.show()

np.savetxt("db_indices_gibbs.txt", db_indices_np)

# %%
angles_R = []

for b in range(B):
    r_b = R_samples_thinned[:, b, :, :].reshape(num_thinned_samples, -1)

    X = X_data[b, :, :] # shape: (d, n)
    s_b = s_samples_thinned[:, b] 
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

np.savetxt("angles_gibbs.txt", np.concatenate(angles_R))

# %%


# %%
