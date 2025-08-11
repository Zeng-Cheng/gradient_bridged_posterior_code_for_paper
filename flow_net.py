# %%
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
import numpy as np
import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import networkx as nx
import random
from scipy.optimize import linprog
from statsmodels.graphics.tsaplots import plot_acf

# %%
# Set seeds for reproducibility
random.seed(246)
np.random.seed(4812)
# JAX uses PRNG keys for randomness; here we set an initial key.
key = jax.random.PRNGKey(123)

# %%
# Define network structure (edges and nodes)
E = jnp.array([
    [0, 1], [2, 4], [3, 6], [4, 6], [5, 6],
    [0, 2], [1, 3], [1, 4], [2, 5], [4, 5]  # Directed edges in the network
])
V = jnp.array([0, 1, 2, 3, 4, 5, 6])  # Set of nodes
s, t = 0, 6  # Define source (s) and sink (t)

# Generate beta (capacity limits) randomly (use float values)
beta_truth = jnp.array([2.9433, 2.8440, 3.2228, 1.2193, 4.0797,
                        6.1826, 4.1528, 2.1969, 2.5417, 2.7563])

# Create a directed graph using networkx. Convert indices to Python ints.
net_graph = nx.DiGraph()
for i, edge in enumerate(np.array(E)):
    u, v = edge
    net_graph.add_edge(int(u), int(v), capacity=float(beta_truth[i]))

# Compute the maximum flow using networkx
flow_value, flow_dict = nx.maximum_flow(net_graph, s, t)
print(f"capacities for edges: {beta_truth}")
print(f"Maximum flow value: {flow_value}")

# Produce the z_truth vector from flow_dict using NumPy first then converting to jnp.array
z_truth_np = np.zeros(E.shape[0])
for i, edge in enumerate(np.array(E)):
    u, v = edge
    z_truth_np[i] = flow_dict[int(u)][int(v)]
z_truth = jnp.array(z_truth_np)
print(f"z vector: {z_truth}")

# %%
# Define free_E and constraint matrices.
free_E = jnp.array([[0, 1], [2, 4], [3, 6], [4, 6], [5, 6]]) 
# shape (5,2) but only number of rows (5) is used below

# Build M1 (shape: (10, 5)) by stacking rows.
row0_5 = jnp.eye(5)  # first 5 rows
row6 = jnp.array([-1, 0, 1, 1, 1])
row7 = jnp.array([0, 0, 1, 0, 0])
row8 = jnp.array([1, 0, -1, 0, 0])
row9 = jnp.array([-1, -1, 1, 1, 1])
row10 = jnp.array([1, 1, -1, -1, 0])
M1 = jnp.vstack([row0_5,
                 row6[None, :],
                 row7[None, :],
                 row8[None, :],
                 row9[None, :],
                 row10[None, :]])  # total 5 + 5 = 10 rows

b1 = beta_truth
M2 = -M1
b2 = jnp.zeros(E.shape[0])
M = jnp.concatenate([M1, M2], axis=0)  # Concatenate constraints (shape: (20, 5))
b = jnp.concatenate([b1, b2])           # Combine constraint values (length 20)

c = np.array([0, 0, -1, -1, -1])
z_lp = linprog(c, A_ub=np.array(M), b_ub=np.array(b)).x
print(np.array(M1) @ z_lp)  # Check if the solution is equal to z_truth

# %%
# Define gradient of h function for flow optimization
@jax.jit
def flow_gradient_h(free_z, M, b, t_log, mu_relu, thres_log):
    term1 = jnp.array([0, 0, 1, 1, 1])
    reg = M.T @ jnp.clip(M @ free_z - b, a_min=0)
    grad_log = (1 / t_log) * M.T @ (1 / jnp.clip(b - M @ free_z, a_min=thres_log))
    grad_h = -term1 + mu_relu * reg + grad_log
    return jnp.sum(grad_h ** 2)

# %%
# Generate synthetic observations y and c with Gaussian noise.
# For JAX randomness, create new keys.
key, subkey1, subkey2 = jax.random.split(key, 3)
sigma_y = 2.0  # Standard deviation for y
sigma_c = 0.5  # Standard deviation for c
ss_y = 1000
ss_c = 1

# Expand beta_truth and z_truth to match dimensions.
c = jax.random.normal(subkey1, shape=(ss_c, beta_truth.shape[0])) * sigma_c + beta_truth
y = jax.random.normal(subkey2, shape=(ss_y, z_truth.shape[0])) * sigma_y + z_truth

# %%
# --------------------------------------------------
# Optimization via JAX's Adam optimizer
# --------------------------------------------------
# this part is used to check the optimization of gradient_h
t_log = 1000
mu_relu = 1000
thres_log = 0.001

b_map = jnp.concatenate([c[0], jnp.zeros(E.shape[0])])
# Define a loss function.
def loss_fn(free_z):
    return flow_gradient_h(free_z, M, b_map, t_log, mu_relu, thres_log)

# Set learning rate and number of steps.
lr = 0.001
num_steps = 10000

# Initialize free_z as the mean of y along axis 0 (take only first 5 elements)
init_params = jnp.mean(y, axis=0)[:free_E.shape[0]]
opt_init, opt_update, get_params = optimizers.adam(lr)
opt_state = opt_init(init_params)

@jax.jit
def step(i, opt_state):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    opt_state = opt_update(i, grads, opt_state)
    return opt_state, loss

for i in range(num_steps):
    opt_state, loss = step(i, opt_state)
    if i % 1000 == 0:
        print(f"Step {i}, Loss: {loss:.5f}")

z_optimized = jnp.dot(M1, get_params(opt_state))
print(f"t_log: {t_log}, mu_relu: {mu_relu}, thres_log: {thres_log}")
print(f"Optimized z: {z_optimized}")
print(f"Optimized sum: {(z_optimized[0] + z_optimized[5]):.3f}")


# %%
# --------------------------------------------------
# Optimization via JAX's Adam optimizer
# --------------------------------------------------
# this part is used to compute the MAP

@jax.jit
def log_posterior(free_z, beta, sigma2_y, sigma2_c, y, c, M, M1,
                  lambda_reg, t_log, mu_relu, thres_log):
    z = jnp.dot(M1, free_z)
    loglik = -jnp.sum((y - z) ** 2) / (2 * sigma2_y)
    loglik -= jnp.sum((c - beta) ** 2) / (2 * sigma2_c)
    loglik -= y.shape[0] * jnp.log(sigma2_y) / 2 + c.shape[0] * jnp.log(sigma2_c) / 2

    b1 = beta
    b2 = jnp.zeros(E.shape[0])
    b = jnp.concatenate([b1, b2])

    reg_term = -lambda_reg * flow_gradient_h(free_z, M, b, t_log, mu_relu, thres_log)
    return loglik + reg_term

# learning_rate = 0.001
# num_steps = 40000
# lambda_reg = 100

# def loss_fn(params):
#     free_z, beta, sigma2_y, sigma2_c = params
#     return -log_posterior(free_z, beta, sigma2_y, sigma2_c, y, c, M, M1,
#                   lambda_reg, t_log, mu_relu, thres_log)

# # Initialize free_z as the mean of y along axis 0 (take only first 5 elements)

# init_params = (jnp.mean(y, axis=0)[:free_E.shape[0]], jnp.mean(c, axis=0),
#                jnp.std(y) ** 2, jnp.std(c) ** 2)
# opt_init, opt_update, get_params = optimizers.adam(lr)
# opt_state = opt_init(init_params)

# @jax.jit
# def step(i, opt_state):
#     params = get_params(opt_state)
#     loss, grads = jax.value_and_grad(loss_fn)(params)
#     opt_state = opt_update(i, grads, opt_state)
#     return opt_state, loss

# for i in range(num_steps):
#     opt_state, loss = step(i, opt_state)
#     if i % 1000 == 0:
#         print(f"Step {i}, Loss: {loss:.5f}")

# z_map, beta_map, sigma2_y_map, sigma2_c_map = get_params(opt_state)
# z_optimized = jnp.dot(M1, z_map)
# print(f"t_log: {t_log}, mu_relu: {mu_relu}, thres_log: {thres_log}")
# print(f"Optimized z: {z_optimized},\n Optimized beta: {beta_map}")
# print(f"Optimized sum: {(z_optimized[0] + z_optimized[5]):.3f}")

# %%
def compute_gradients(free_z, beta, M, t_log, mu_relu, thres_log):
    b = jnp.concatenate([beta, jnp.zeros(beta.shape[0])])  # Construct b from beta
    v = M @ free_z - b
    
    # Compute indicator functions
    indicator_v_pos = (v > 0).astype(jnp.float32)
    indicator_v_neg = (-v > thres_log).astype(jnp.float32)
    
    # Compute the diagonal matrices
    D1 = jnp.diag(indicator_v_pos)
    D2 = jnp.diag(indicator_v_neg / (v**2 + 1e-8))
    
    P_zero = jnp.zeros((beta.shape[0], beta.shape[0]))
    P = jnp.concatenate([jnp.eye(beta.shape[0]), P_zero])

    # Compute the gradients explicitly
    df_dz = mu_relu * M.T @ D1 @ M + (1 / t_log) * M.T @ D2 @ M
    df_dbeta = -mu_relu * M.T @ D1 @ P - (1 / t_log) * M.T @ D2 @ P
    
    return jnp.vstack([df_dz, df_dbeta.T, jnp.zeros(free_z.shape[0]), jnp.zeros(free_z.shape[0])])

G = compute_gradients(jnp.mean(y, axis=0)[:free_E.shape[0]], jnp.mean(c, axis=0),
                      M, t_log, mu_relu, thres_log)
pj_term = G @ jnp.linalg.inv(G.T @ G) @ G.T
inverse_mass = (1 + 0.001) * jnp.eye(G.shape[0]) - pj_term

# %%
# -----------------------------------
# This is using h rather than gradient-h
# -----------------------------------
# @jax.jit
# def flow_h(free_z, M, b, t_log, mu_relu, thres_log):
#     term1 = free_z[2] + free_z[3] + free_z[4]
#     reg = jnp.sum(jnp.clip(M @ free_z - b, a_min=0) ** 2)
#     # grad_log = (1 / t_log) * jnp.sum(jnp.log(jnp.clip(b - M @ free_z, a_min=thres_log)))
#     h = -term1 + mu_relu * reg / 2 # - grad_log
#     return h

# @jax.jit
# def log_posterior(free_z, beta, sigma2_y, sigma2_c, y, c, M, M1,
#                   lambda_reg, t_log, mu_relu, thres_log):
#     z = jnp.dot(M1, free_z)
#     loglik = -jnp.sum((y - z) ** 2) / (2 * sigma2_y)
#     loglik -= jnp.sum((c - beta) ** 2) / (2 * sigma2_c)
#     loglik -= y.shape[0] * jnp.log(sigma2_y) / 2 + c.shape[0] * jnp.log(sigma2_c) / 2

#     b1 = beta
#     b2 = jnp.zeros(E.shape[0])
#     b = jnp.concatenate([b1, b2])

#     reg_term = -lambda_reg * flow_h(free_z, M, b, t_log, mu_relu, thres_log)
#     return loglik + reg_term

# %%
# Define probabilistic model for Bayesian inference using numpyro.
def model(y, c, M, M1, lambda_reg, t_log, mu_relu, thres_log):
    beta = numpyro.sample("beta", dist.HalfNormal(10.0).expand([E.shape[0]]))
    free_z = numpyro.sample("free_z", dist.HalfNormal(10.0).expand([free_E.shape[0]]))
    sigma2_y = numpyro.sample("sigma2_y", dist.InverseGamma(2.0, 5.0))
    sigma2_c = numpyro.sample("sigma2_c", dist.InverseGamma(5.0, 2.0))
    
    log_prob = log_posterior(free_z, beta, sigma2_y, sigma2_c, y, c, M, M1,
                  lambda_reg, t_log, mu_relu, thres_log)
    
    numpyro.factor("log_prob", log_prob)

# Set seed for reproducibility
key = jax.random.PRNGKey(50)
lambda_reg = 200

# Initialize parameters using MAP estimates.
initial_params = {'free_z': jnp.mean(y, axis=0)[:free_E.shape[0]],
                  'beta': jnp.mean(c, axis=0),
                  'sigma2_y': jnp.std(y) ** 2,
                  'sigma2_c': jnp.std(c) ** 2}

# Perform MCMC sampling using NumPyro's NUTS.
# nuts_kernel = NUTS(
#     model, init_strategy=init_to_value(values=initial_params),
#     dense_mass=[("free_z", "beta", "sigma2_y", "sigma2_c")],
#     inverse_mass_matrix={("free_z", "beta", "sigma2_y", "sigma2_c"): inverse_mass})
nuts_kernel = NUTS(model, init_strategy=init_to_value(values=initial_params))
mcmc = MCMC(nuts_kernel, num_samples=10000, num_warmup=2000)
mcmc.run(key, y, c, M, M1, lambda_reg, t_log, mu_relu, thres_log)
samples = mcmc.get_samples()

# collect samples
z_samples = np.array(jnp.dot(samples['free_z'], M1.T))
beta_samples = np.array(samples['beta'])
print(f"z_samples: {z_samples.shape}, beta_samples: {beta_samples.shape}")

# %%
# ===========================
# save the posterior samples
# ===========================

# change the method name for different methods
method = 'grad-bridge-nomass'
# grad-bridge; kernel-h; gibbs; grad-bridge-nomass

np.savetxt('res/flow_net_beta_samples_' + method + '.txt', beta_samples, delimiter=',')
np.savetxt('res/flow_net_z_samples_' + method + '.txt', z_samples, delimiter=',')
sigma2_c_samples = np.array(samples['sigma2_c'])
np.savetxt('res/flow_net_sigma2_c_samples_' + method + '.txt', sigma2_c_samples, delimiter=',')
sigma2_y_samples = np.array(samples['sigma2_y'])
np.savetxt('res/flow_net_sigma2_y_samples_' + method + '.txt', sigma2_y_samples, delimiter=',')

# %%

# Plotting for traces and autocorrelation
plt.figure(figsize=(15, 30))
for i in range(10):
    plt.subplot(10, 2, 2 * i + 1)
    z_component = z_samples[:, i]
    plt.plot(z_component)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'$z_{i}$', fontsize=14)
    
    plt.subplot(10, 2, 2 * i + 2)
    ax = plt.gca()
    plot_acf(z_component, lags=40, ax=ax, title=None, auto_ylims=True)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Autocorrelation', fontsize=14)

plt.tight_layout(pad=3)
plt.show()

# %%
# Plotting for traces and autocorrelation
plt.figure(figsize=(15, 30))
for i in range(10):
    plt.subplot(10, 2, 2 * i + 1)
    beta_component = beta_samples[:, i]
    plt.plot(beta_component)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'$beta_{i}$', fontsize=14)
    
    plt.subplot(10, 2, 2 * i + 2)
    ax = plt.gca()
    plot_acf(beta_component, lags=40, ax=ax, title=None, auto_ylims=True)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('Autocorrelation', fontsize=14)

plt.tight_layout(pad=3)
plt.show()

# %%
# ======================
# histgrams of samples
# ======================

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.hist(z_samples[:, i], bins=30, density=True, alpha=0.7)
    plt.axvline(np.array(z_truth)[i], color='red', linestyle='--', linewidth=2)
    plt.xlabel(f"$z_{i}$", fontsize=14)
    plt.ylabel("Density", fontsize=14)
plt.tight_layout(pad=3)
plt.show()

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.hist(beta_samples[:, i], bins=30, density=True, alpha=0.7)
    plt.axvline(np.array(beta_truth)[i], color='red', linestyle='--', linewidth=2)
    plt.xlabel(f"$\\beta_{i}$", fontsize=14)
    plt.ylabel("Density", fontsize=14)
plt.tight_layout(pad=3)
plt.show()
# %%
