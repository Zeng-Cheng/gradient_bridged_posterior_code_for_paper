# %%
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS
from numpyro.infer.initialization import init_to_value
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pickle
from tqdm.notebook import trange

# %%
# --------------------------------------------------
# Data loading
# --------------------------------------------------
file_path = "generated_data.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

x = jnp.array(data['x']).T[0]
# Convert to NumPy for plotting if needed
x_np = np.array(data['x'].squeeze())
z_truth = jnp.array(data['z'])
y = jnp.array(data['y'])
p = x.shape[0]

# %%
# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def precompute_base_matrix(x, b):
    x_diff = jnp.subtract.outer(x, x)  # shape (p, p)
    base_matrix = jnp.exp(-x_diff ** 2 / (2 * b))
    return base_matrix + 0.01 * jnp.eye(p)

@jax.jit
def log_posterior(w, tau, b, x, y, lambda_reg=100):
    Q = precompute_base_matrix(x, b) * tau
    alpha = 1 - 1 / (1 + jnp.exp(w)) - y
    z = - Q @ alpha
    term1 = jnp.sum(y * z)
    term2 = -jnp.sum(jnp.logaddexp(0.0, z))
    logL = -0.5 * jnp.dot(alpha, Q @ alpha) + term1 + term2
    gradient_g = Q @ alpha + w
    logreg = -lambda_reg * jnp.sum(gradient_g ** 2)
    return logL + logreg


# %%
# --------------------------------------------------
# Optimization via JAX's Adam optimizer
# --------------------------------------------------
learning_rate = 0.001
num_steps = 20000
lambda_reg = 100

tau_map = jnp.array([2.3])
b_map = jnp.array([0.3])

def loss_fn(params):
    w = params
    Q = precompute_base_matrix(x, b_map) * tau_map
    alpha = 1 - 1 / (1 + jnp.exp(w)) - y
    gradient_g = Q @ alpha + w
    return jnp.sum(gradient_g ** 2)

# Initialize parameters: w as zeros, tau and b as scalars
init_params = jnp.zeros(p)
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(init_params)

@jax.jit
def step(i, opt_state):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    opt_state = opt_update(i, grads, opt_state)
    return opt_state, loss

for i in trange(num_steps):
    opt_state, loss = step(i, opt_state)
    if i % 1000 == 0:
        print(f"Step {i}, Loss: {loss:.5f}")

w_map = get_params(opt_state)
Q = precompute_base_matrix(x, b_map) * tau_map
alpha = 1 - 1 / (1 + jnp.exp(w_map)) - y
z_opt = - Q @ alpha

# Plot optimized z versus truth
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(x_np, np.array(z_opt), color='blue')
plt.xlabel('x')
plt.ylabel('z')
plt.subplot(1, 2, 2)
plt.plot(x_np, np.array(z_truth), color='orange')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout(pad=3)
plt.show()

# %%

def compute_gradients(w, tau, b, x, y):
    """
    Computes the vector function h_\alpha's derivatives with respect to alpha, tau, and b.
    f is defined for each i as:
        f_i = sum_{j=1}^n Q_{ij} * alpha_j + log((alpha_i + y_i)/(1 - alpha_i - y_i))
    
    where:
        Q_{ij} = tau * exp(-||X_i - X_j||^2/(2b))

    Returns:
      dfdalpha  : (n, n) Jacobian matrix of f w.r.t. alpha.
      dfdtau    : (n,) vector, derivative of f w.r.t tau.
      dfdb      : (n,) vector, derivative of f w.r.t b.
    """

    alpha = 1 - 1 / (1 + jnp.exp(w)) - y

    """
    Computes the n x n matrix E with entries:
    E_{ij} = exp(-||X_i - X_j||^2/(2b))
    """
    # Assuming X is (n, )
    diff = x[:, None] - x[None, :]  # shape (n, n)
    E = jnp.exp(-diff**2 / (2 * b)) + 0.01 * jnp.eye(p)

    """
    Computes the n x n matrix F with entries:
    F_{ij} = tau * exp(-||X_i - X_j||^2/(2b)) * (||X_i - X_j||^2/(2b^2))
    """
    F = tau * E * (diff**2 / (2 * b**2))

    # Compute Q matrix:
    Q = tau * E # shape (n, n)

    # Compute derivative of f with respect to w
    sig_deriv = (alpha + y) * (1 - alpha - y)
    Jw = Q * sig_deriv[None, :]
    dfdalpha = Jw + jnp.eye(p)
    
    # # Compute derivative of f with respect to alpha.
    # diag_deriv = 1 / (alpha + y) + 1 / (1 - alpha - y)  # shape (n,)
    # # Build the full Jacobian (n x n)
    # dfdalpha = Q + jnp.diag(diag_deriv)
    
    # Derivative with respect to tau:
    # dQ_{ij}/dtau = exp(-||X_i-X_j||^2/(2b))
    dfdtau = E @ alpha       # shape (n,)
    
    # Derivative with respect to b:
    # dQ_{ij}/db = tau * exp(-||X_i-X_j||^2/(2b)) * (||X_i-X_j||^2/(2b^2))
    dfdb = F @ alpha         # shape (n,)
    
    return jnp.vstack([dfdalpha, dfdtau, dfdb])

G = compute_gradients(w_map, tau_map, b_map, x, y)
pj_term = G @ jnp.linalg.inv(G.T @ G) @ G.T
inverse_mass = (1 + 0.001) * jnp.eye(G.shape[0]) - pj_term

# %%
def model(x, y, lambda_reg):
    w = numpyro.sample("w", dist.Normal(0.0, 10.0).expand([p])) # type: ignore
    tau = numpyro.sample("tau", dist.InverseGamma(2.0, 5.0))
    b = numpyro.sample("b", dist.InverseGamma(2.0, 5.0))
    logL = log_posterior(w, tau, b, x, y, lambda_reg)
    numpyro.factor("log_prob", logL)

# %%
# --------------------------------------------------
# MCMC using NumPyro
# --------------------------------------------------
# Use current estimates as initial values
initial_params = {'w': w_map, 'tau': tau_map, 'b': b_map}
nuts_kernel = HMC(
    model, init_strategy=init_to_value(values=initial_params),
    dense_mass=[("w", "tau", "b")], # type: ignore
    inverse_mass_matrix={("w", "tau", "b"): inverse_mass})
# nuts_kernel = NUTS(model, init_strategy=init_to_value(values=initial_params))
mcmc = MCMC(nuts_kernel, num_samples=12000, num_warmup=2000)
rng_key = jax.random.PRNGKey(70)
mcmc.run(rng_key, x, y, lambda_reg)
samples = mcmc.get_samples()



# %%
w_samples = samples['w']
tau_samples = samples['tau']
b_samples = samples['b']



# %%
# Compute z for each sample
alpha_samples = 1 - 1 / (1 + jnp.exp(w_samples)) - y  # shape (num_samples, p)
num_mcmc = w_samples.shape[0]
z_samples = []
for i in range(num_mcmc):
    base_matrix = precompute_base_matrix(x, b_samples[i])
    z_val = -tau_samples[i] * (base_matrix @ alpha_samples[i])
    z_samples.append(z_val)
z_samples = jnp.stack(z_samples)



# %%
# --------------------------------------------------
# Plotting MCMC results for w, tau and b
# --------------------------------------------------
w_samples_np = np.array(w_samples)
plt.figure(figsize=(15, 30))
for i in range(10):
    plt.subplot(10, 2, 2*i+1)
    plt.plot(w_samples_np[:, i])
    plt.xlabel('Iteration')
    plt.ylabel(f'w_{i}')
    plt.subplot(10, 2, 2*i+2)
    ax = plt.gca()
    plot_acf(w_samples_np[:, i], lags=40, ax=ax, title=None, auto_ylims=True) # type: ignore
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
plt.tight_layout(pad=3)
plt.show()

# %%
tau_samples_np = np.array(tau_samples)
b_samples_np = np.array(b_samples)

plt.figure(figsize=(10, 5))
# Plot trace and ACF for tau
# Traceplot for tau
plt.subplot(2, 2, 1)
plt.plot(tau_samples_np)
plt.xlabel('Iteration')
plt.ylabel('tau')
# ACF plot for tau
plt.subplot(2, 2, 2)
ax = plt.gca()
plot_acf(tau_samples_np, lags=40, ax=ax, title=None, auto_ylims=True) # type: ignore
plt.xlabel('Lag')
plt.ylabel('tau')
# Plot trace and ACF for b
# Traceplot for tau
plt.subplot(2, 2, 3)
plt.plot(b_samples_np)
plt.xlabel('Iteration')
plt.ylabel('b')
# ACF plot for tau
plt.subplot(2, 2, 4)
ax = plt.gca()
plot_acf(b_samples_np, lags=40, ax=ax, title=None, auto_ylims=True) # type: ignore
plt.xlabel('Lag')
plt.ylabel('b')
plt.tight_layout(pad=3)
plt.show()

# %%

posterior_mean_z = jnp.mean(z_samples[::20, :], axis=0)
posterior_mean_w = jnp.mean(w_samples[::20, :], axis=0)

plt.figure(figsize=(5, 3))
plt.scatter(np.array(posterior_mean_z), np.array(posterior_mean_w), s=0.5)
plt.xlabel('z')
plt.ylabel('w')
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(x_np, np.array(posterior_mean_z), color='blue')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(x_np, np.array(z_truth), color='orange')
plt.xlabel('x')
plt.ylabel('z')
plt.show()
# %%
