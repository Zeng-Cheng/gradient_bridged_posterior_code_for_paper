### Gradient-bridged Posterior: Bayesian Inference for Models with Implicit Functions

This repository contains the reproducibility materials for the paper: Gradient-bridged Posterior: Bayesian Inference for Models with Implicit Functions.

#### Abstract

Many statistical problems include model parameters that are defined as the solutions to optimization sub-problems. These include classical approaches such as profile likelihood as well as  modern applications involving flow networks or Procrustes distances. In such cases, the likelihood of the data involves an implicit function, often complicating inferential procedures and entailing prohibitive computational cost. In this article, we propose an intuitive and tractable posterior inference approach for this setting. We introduce a class of continuous models that handle implicit function values using the first-order optimality of the sub-problems. Specifically, we apply a shrinkage kernel to the gradient norm, which retains a probabilistic interpretation within a generative model. This can be understood as a generalization of the Gibbs posterior framework to newly enable concentration around partial minimizers in a subset of the parameters. We show that this method, termed the gradient-bridged posterior, is amenable to efficient posterior computation, and enjoys theoretical guarantees, establishing a Bernstein--von Mises theorem for asymptotic normality. The advantages of our approach  are highlighted on a synthetic flow network experiment and an application to data integration using Procrustes distances.

#### Packages needed in the repository

- Python packages: torch, numpy, jax, numpyro, pyro, scipy, tqdm, polyagamma
- R packages: ggplot2, coda, shapes

#### Guidance of using the codes to reproduce the figures and tables in the paper

- Figure 1 is not produced by codes.
- Results in Section 5.
    - First, in `flow-net-numpyro.py`, we sample some simulated data, run MCMC to produce the posterior samples, and save the samples in `res/`.
    - Figure 3 and 4 are plotted by `plot-flow-net.R`.
- Results in Section 6.
    - `data_preprocessing.R`
    - `data_application_numpyro.py`