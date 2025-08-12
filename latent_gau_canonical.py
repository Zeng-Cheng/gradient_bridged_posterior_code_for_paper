# %%
from tqdm.notebook import trange
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import numpy as np
import scipy
from polyagamma import random_polyagamma
import pickle
import matplotlib.pyplot as plt


def latent_gau_log_prob(zeta, dist_x, y, tau, b, alpha_prior = 2, beta = 5):
    """
    Args:
        zeta: the latent Gaussian variables; vector of 1*n
        dist_x: matrix of |xi-xj|^2
    """
    if tau <= 0:
        raise ValueError("tau must be greater than 0")
    
    if b <= 0:
        raise ValueError("b must be greater than 0")
    
    n = y.size(0) # sample size
    q_matrix = (torch.exp(-dist_x / b / 2)).float() + 0.01 * torch.eye(n)
    # q_matrix is without tau, and we add eps*I to solve singular problem

    log_tau_prior = -tau ** 2 / 2
    log_b_prior = (-alpha_prior - 1) * torch.log(b) - beta / b
    loglik = (-zeta @ torch.linalg.solve(q_matrix, zeta) / 2 / tau
                - torch.logdet(q_matrix) / 2 - n * torch.log(tau) / 2)

    if torch.isnan(loglik):
        return float('-inf')
    return loglik + log_tau_prior + log_b_prior


def acc_rate(accept, i, lags = 200):
    if i < lags:
        return np.sum(accept[:i]) / i
    else:
        return np.sum(accept[(i - lags):i]) / lags
    
# %% 
file_path = "generated_data.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

X = data['x']
dist_x = torch.cdist(X, X) ** 2
z_truth = data['z']
y = data['y']
n = X.shape[0]

# %%

alpha_prior = 2
beta = 5

# simulating parameters
num_samples = 10000
burn_in = 2000
lam_tau = 0.5
lam_b = 0.5

tau_aug_trace, b_aug_trace = [], []
accept_tau, accept_b = np.zeros(num_samples), np.zeros(num_samples)

# initial values
tau = torch.tensor([1.5])
b = torch.tensor([1.5])
eta = torch.tensor(random_polyagamma(1, size=n))

for ns in trange(num_samples):
    
    # Gibbs step via data augmentation

    q_matrix = (torch.exp(-dist_x / b / 2)).float() + 0.01 * torch.eye(n)
    Omega = torch.diag(eta)
    w_var = torch.inverse(torch.inverse(q_matrix) / tau + Omega).float()
    w_var = (w_var + w_var.T) / 2
    w_mean = w_var @ (y - 0.5)
    w = MVN(w_mean, w_var).sample()
    eta = torch.tensor(random_polyagamma(1, w))
    logprob = latent_gau_log_prob(w, dist_x, y, tau, b, alpha_prior, beta)
    
    # Metroplis--Hastings

    tau_prop = tau + torch.randn(1) * lam_tau
    if (tau_prop > 0): 
        logprob_prop = latent_gau_log_prob(w, dist_x, y, tau_prop, b, alpha_prior, beta)
        if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
            tau = tau_prop
            logprob = logprob_prop
            accept_tau[ns] = 1
        
    b_prop = b + torch.randn(1) * lam_b
    if b_prop > 0:
        logprob_prop = latent_gau_log_prob(w, dist_x, y, tau, b_prop, alpha_prior, beta)
        if (torch.log(torch.rand(1)) < (logprob_prop - logprob)):
            b = b_prop
            logprob = logprob_prop
            accept_b[ns] = 1
            
    # adjust the step size of random walk
            
    if (ns % 200 == 0) and (ns < burn_in):
        if acc_rate(accept_tau, ns + 1) > 0.4:
            lam_tau *= 2
        if acc_rate(accept_tau, ns + 1) < 0.3:
            lam_tau /= 3
        if acc_rate(accept_b, ns + 1) > 0.4:
            lam_b *= 2
        if acc_rate(accept_b, ns + 1) < 0.3:
            lam_b /= 3
    tau_aug_trace.append(tau)
    b_aug_trace.append(b)
    if ns % 1000 == 0:
        print('step:, {:d}, accept_rate of tau: {:.3f}, of b: {:.3f}'.format(
            ns, acc_rate(accept_tau, ns + 1), acc_rate(accept_b, ns + 1)))

tau_aug_trace, b_aug_trace = np.stack(tau_aug_trace), np.stack(b_aug_trace)


# %%
fig, ax = plt.subplots(1, 5, figsize = (14, 3))
ax[0].set_title("Trace for tau")
ax[0].plot(tau_aug_trace)

ax[2].set_title("Trace for b")
ax[2].plot(b_aug_trace)

ax[4].set_title("Trace for tau after burnin")
ax[4].plot(tau_aug_trace[burn_in:])

ax[3].set_title("Trace for b after burnin")
ax[3].plot(b_aug_trace[burn_in:])

ax[1].set_title("Trace plot after burnin")
ax[1].scatter(tau_aug_trace[burn_in:], b_aug_trace[burn_in:], s=0.6)


# %%
