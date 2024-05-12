# Astronomical linear simulation based inference
Code and plots for the NST Physics Part III project "Astronomical Linear Simulation-based Inference"



## Abstract

We develop the theoretical framework of Linear Simulation-based Inference (LSBI), an application of likelihood-free inference where the model is approximated by a linear function of its parameters and the noise is assumed to be Gaussian with zero mean. We obtain analytical expressions for the posterior distributions of hyperparameters of the linear likelihood in terms of samples drawn from a simulator. This method is applied to several toy models, and to emulated datasets for the Cosmic Microwave Background power spectrum and the sky-averaged 21cm hydrogen line. We find that convergence is achieved after  $\mathcal{O}(10^4)$ simulations at most. Furthermore, LSBI is coupled with massive linear compression into a set of $n$ summary statistics, where $n$ is the number of parameters of the model, reducing the computational time by two to three orders of magnitude.  Therefore, we demonstrate that it is possible to obtain significant information gain and generate posteriors that agree with the underlying parameters while maintaining explainability and intellectual oversight.
