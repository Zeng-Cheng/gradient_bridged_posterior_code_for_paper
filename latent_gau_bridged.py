# %%
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, lax, device_put
import scipy
import pickle
import matplotlib.pyplot as plt
from tqdm.notebook import trange

def latent_gau_optim_step(dist_x, y, tau, b):
    q_matrix = tau * (jnp.exp(-dist_x / 2 / b) + 0.01 * jnp.eye(n))
    # z is alpha + y \in (0, 1)
    logit_z = jnp.zeros(n)
    def cal_loss(logit_z):
        z = 1 - 1 / (1 + jnp.exp(logit_z))
        res = z * jnp.log(z / (1 - z)) + jnp.log(1 - z)
        loss = (z - y) @ q_matrix @ (z - y) / 2 + res.sum()
        return loss

    for _ in range(40):
        loss = cal_loss(logit_z)
        
        logit_z_grad = grad(cal_loss)(logit_z)
        step = jnp.array([0.5])
        def cond_fun(step):
            return (cal_loss(logit_z - step * logit_z_grad) > loss - 
                step * jnp.linalg.norm(logit_z_grad) ** 2 / 2)[0]
        def body_fun(step):
            return step * 0.6
        step = lax.while_loop(cond_fun, body_fun, step)
        logit_z -= step * logit_z_grad
    
    z = 1 - 1 / (1 + jnp.exp(logit_z))
    return -q_matrix @ (z - y), z - y

optim_jit = jit(latent_gau_optim_step)
    
def acc_rate(accept, i, lags = 200):
    if i < lags:
        return np.sum(accept[:i]) / i
    else:
        return np.sum(accept[(i - lags):i]) / lags

# %%
file_path = "data/generated_data.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

X = data['x'].numpy()
dist_x = scipy.spatial.distance.cdist(X, X) ** 2
z_truth = data['z'].numpy()
y = data['y'].numpy()
n = X.shape[0]


# %%
alpha_prior = 2
beta = 5
# sampling parameters
num_samples = 10000
burn_in = 2000

lam_tau = 1
lam_b = 0.5

zeta_trace, tau_trace, b_trace, logp_trace = [], [], [], []
accept_tau, accept_b = np.zeros(num_samples), np.zeros(num_samples)

# initial values
tau = jnp.array([2])
b = jnp.array([1.5])

zeta, alpha = optim_jit(dist_x, y, tau, b)
q_matrix = tau * (jnp.exp(-dist_x / 2 / b) + 0.01 * jnp.eye(n))
log_tau_prior = (-alpha_prior - 1) * jnp.log(tau) - (beta / tau)
log_b_prior = (-alpha_prior - 1) * jnp.log(b) - (beta / b)
res_latent_gau = -alpha @ q_matrix @ alpha / 2

res_logistic_p = y * zeta - jnp.logaddexp(jnp.zeros(1), zeta)
loglik = res_latent_gau + res_logistic_p.sum()

if jnp.isnan(loglik):
    loglik = float('-inf')
logprob = loglik + log_tau_prior + log_b_prior

for ns in trange(num_samples):
    
    tau_prop = device_put(tau + np.random.randn(1) * lam_tau)
    if (tau_prop > 0):
        zeta_prop, alpha_prop = optim_jit(dist_x, y, tau_prop, b)
        q_matrix = tau_prop * (jnp.exp(-dist_x / 2 / b) + 0.01 * jnp.eye(n))
        log_tau_prior = (-alpha_prior - 1) * jnp.log(tau_prop) - (beta / tau_prop)
        log_b_prior = (-alpha_prior - 1) * jnp.log(b) - (beta / b)
        res_latent_gau = -alpha_prop @ q_matrix @ alpha_prop / 2
        
        res_logistic_p = y * zeta_prop - jnp.logaddexp(jnp.zeros(1), zeta_prop)
        loglik = res_latent_gau + res_logistic_p.sum()
        
        if jnp.isnan(loglik):
            loglik = float('-inf')
        logprob_prop = loglik + log_tau_prior + log_b_prior
        
        if (jnp.log(np.random.rand(1)) < (logprob_prop - logprob)):
            tau = tau_prop
            zeta = zeta_prop
            alpha = alpha_prop
            logprob = logprob_prop
            accept_tau[ns] = 1
            
    b_prop = device_put(b + np.random.randn(1) * lam_b)
    if b_prop > 0:
        zeta_prop, alpha_prop = optim_jit(dist_x, y, tau, b_prop)
        q_matrix = tau * (jnp.exp(-dist_x / 2 / b_prop) + 0.01 * jnp.eye(n))
        log_tau_prior = (-alpha_prior - 1) * jnp.log(tau) - (beta / tau)
        log_b_prior = (-alpha_prior - 1) * jnp.log(b_prop) - (beta / b_prop)
        res_latent_gau = -alpha_prop @ q_matrix @ alpha_prop / 2
        
        res_logistic_p = y * zeta_prop - jnp.logaddexp(jnp.zeros(1), zeta_prop)
        loglik = res_latent_gau + res_logistic_p.sum()
        
        if jnp.isnan(loglik):
            loglik = float('-inf')
        logprob_prop = loglik + log_tau_prior + log_b_prior
        
        if (jnp.log(np.random.rand(1)) < (logprob_prop - logprob)):
            b = b_prop
            zeta = zeta_prop
            alpha = alpha_prop
            logprob = logprob_prop
            accept_b[ns] = 1

    if (ns % 200 == 0) and (ns < burn_in):
        if acc_rate(accept_tau, ns + 1) > 0.5:
            lam_tau *= 1.5
        if acc_rate(accept_tau, ns + 1) < 0.25:
            lam_tau /= 3.5
        if acc_rate(accept_b, ns + 1) > 0.5:
            lam_b *= 1.5
        if acc_rate(accept_b, ns + 1) < 0.25:
            lam_b /= 3.5
        
    zeta_trace.append(zeta)
    tau_trace.append(tau)
    b_trace.append(b)
    logp_trace.append(logprob)
    if ns % 400 == 0:
        print('Step: {:d}, Accept_rate of tau: {:.3f}, Accept_rate of b: {:.3f}'.format(
            ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))
zeta_trace, logp_trace = np.stack(zeta_trace), np.stack(logp_trace)
tau_trace, b_trace = np.stack(tau_trace), np.stack(b_trace)

# %%
fig, ax = plt.subplots(2, 4, figsize = (14, 6))
ax[0, 0].set_title("Trace for tau")
ax[0, 0].plot(tau_trace)

ax[0, 2].set_title("Trace for b")
ax[0, 2].plot(b_trace)

for i in range(50):
    ax[0, 1].violinplot(dataset=zeta_trace[burn_in:, i], positions=[i])
ax[0, 1].set_title("Posterior Distribution of w")

ax[0, 3].set_title("Posterior Mean of W vs X")
ax[0, 3].scatter(X, zeta_trace[burn_in:, ].mean(0), s=0.6)

ax[1, 0].set_title("Trace for tau after burnin")
ax[1, 0].plot(tau_trace[burn_in:])

ax[1, 2].set_title("Trace for b after burnin")
ax[1, 2].plot(b_trace[burn_in:])

ax[1, 1].set_title("Trace plot after burnin")
ax[1, 1].scatter(tau_trace[burn_in:], b_trace[burn_in:], s=0.6)

ax[1, 3].set_title("Trace for logprob after burnin")
ax[1, 3].plot(logp_trace[burn_in:])
# %%
