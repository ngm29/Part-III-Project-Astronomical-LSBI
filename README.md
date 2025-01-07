# Astronomical linear simulation based inference
Code and plots for the NST Physics Part III project "Astronomical Linear Simulation-based Inference"

Web reference: [Galileo -  Part III Projects supervised by Will Handley](https://www.mrao.cam.ac.uk/~wh260/Galileo)



## Abstract

We develop the framework of Linear Simulation-based Inference (LSBI), an application of simulation-based inference where the likelihood is approximated by a Gaussian linear function of its parameters. We obtain analytical expressions for the posterior distributions of hyper-parameters of the linear likelihood in terms of samples drawn from a simulator, for both uniform and conjugate priors.  This method is applied sequentially to several toy-models and tested on emulated datasets for the Cosmic Microwave Background temperature power spectrum. We find that convergence is achieved after four or five rounds of $\mathcal{O}(10^4)$ simulations, which is competitive with state-of-the-art neural density estimation methods. Therefore, we demonstrate that it is possible to obtain significant information gain and generate posteriors that agree with the underlying parameters while maintaining explainability and intellectual oversight.

## Dependencies

* Python 3.9+
* lsbi - [GitHub](https://github.com/handley-lab/lsbi/tree/master), [Documenation](https://lsbi.readthedocs.io/en/latest/)
* globalemu - [GitHub](https://github.com/htjb/globalemu), [Documenation](https://globalemu.readthedocs.io/en/latest/)
* cmbemu - [GitHub](https://github.com/htjb/cmbemu/tree/main)
* getdist - [GitHub](https://github.com/cmbant/getdist), [Documentation](https://getdist.readthedocs.io/en/latest/)

## Citations

Handley et al, (2024) lsbi: Linear Simulation Based Inference.

Bevins, H., Handley, W. J., Fialkov, A., Acedo, E. D. L., and Javid, K. (2021). GLOBALEMU: A novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionisation. arXiv:2104.04336

Lewis, A. (2019). GetDist: a Python package for analysing Monte Carlo samples. arXiv preprint arXiv:1910.13970.
