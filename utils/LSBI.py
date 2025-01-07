import numpy as np
from scipy.stats import invwishart
from lsbi.model import LinearModel



def LSBI(θ, D, D_0, C_0=0, ν_0=0, Θ_0=0, *args, **kwargs):
    '''
    Implementation of Linear Simulation-Based Inference (LSBI).
    Here, n denotes the number of parameters, d is the dimension of the dataset, and k is the number of simulated parameter-data pairs

    Parameters
    ----------
    θ : array-like, dimensions (k,n)
        Vector of parameter samples for each simulator run
    D : array-like, dimensions (k,d)
        Vector of data samples for each simulator run
    D_0 : array-like, dimensions (d)
        Vector of observed data
    C_0 : array-like, dimensions (d,d)
        Prior for the covariance matrix C; default = 0
    ν_0 : int
        Prior degrees of freedom; default = 0
    Θ_0 : array-like, dimensions (n,n)
        Prior for model matrix M (can be different to the prior covariance), defualt = 0
    shape : int or tuple
        Number of samples drawn from posterior distributions
        Can be made a tuple to get an array of custom dimensions

    Returns
    -------
    model: LinearModel object with updated likelihood hyper-parameters
    samples: posterior samples drawn form the model
    logpdf: log-probability of the samples
    '''

    shape = kwargs.pop('shape', ())
    if isinstance(shape, int):
        shape = (1,1,shape)
    k, n = θ.shape
    d = D.shape[1]
    if ν_0 < d:
        ν_0 = d

    N_p = 100000
    N = np.prod(shape)

    θD = np.concatenate([θ, D], axis=1)
    mean = θD.mean(axis=0)
    θbar = mean[:n]
    Dbar = mean[n:]

    cov = np.cov(θD.T)
    Θ = cov[:n, :n]
    Δ = cov[n:, n:]
    Ψ = cov[n:, :n]
    ν = k - d - n - 2
    invΘ = np.linalg.inv(Θ + Θ_0 / (k - 1))

    C_ = invwishart(df=ν, scale=((k - 1) * (Δ - Ψ @ invΘ @ Ψ.T) + C_0)).rvs(size=shape[2])
    C_ = C_[np.newaxis,np.newaxis,:]
    C_ = np.repeat(np.repeat(C_,shape[1], axis=1),shape[0],axis=0)

    L1 = np.linalg.cholesky(C_[0])/ np.sqrt(k-1)
    L2 = np.linalg.cholesky(invΘ)
    L3 = np.linalg.cholesky(C_) / np.sqrt(k+1)

    M_ = Ψ @ invΘ + np.einsum('...jk,...kl,ml->...jm', L1, np.random.randn(shape[1],shape[2],d, n), L2)
    M_ = M_[np.newaxis,:]
    M_ = np.repeat(M_, shape[0], axis=0)

    m_ = (k / (k + 1)) * (Dbar - M_ @ θbar) + np.einsum('...jk,...k->...j', L3, np.random.randn(*shape,d))

    model = LinearModel(m=m_, M=M_, C=C_, *args, **kwargs)

    samples = model.posterior(D_0).rvs(size=N_p // N).reshape((N_p//N)*N, n)

    logpdf = np.log(model.posterior(D_0).pdf(samples).mean(axis=(-1,-2,-3))).reshape((N_p//N)*N)

    return model, samples, logpdf


def LSBI_diag(θ, D, D_0, C_0=0, ν_0=0, Θ_0=0, *args, **kwargs):
    '''
    Implementation of Linear Simulation-Based Inference (LSBI).
    Here, n denotes the number of parameters, d is the dimension of the dataset, and k is the number of simulated parameter-data pairs

    Parameters
    ----------
    θ : array-like, dimensions (k,n)
        Vector of parameter samples for each simulator run
    D : array-like, dimensions (k,d)
        Vector of data samples for each simulator run
    D_0 : array-like, dimensions (d)
        Vector of observed data
    C_0 : array-like, dimensions (d,d)
        Prior for the covariance matrix C; default = 0
    ν_0 : int
        Prior degrees of freedom; default = 0
    Θ_0 : array-like, dimensions (n,n)
        Prior for model matrix M (can be different to the prior covariance), defualt = 0
    shape : int or tuple
        Number of samples drawn from posterior distributions
        Can be made a tuple to get an array of custom dimensions

    Returns
    -------
    model: LinearModel object with updated likelihood hyper-parameters
    samples: posterior samples drawn form the model
    logpdf: log-probability of the samples
    '''

    shape = kwargs.pop('shape', ())
    if isinstance(shape, int):
        shape = (1, 1, shape)
    k, n = θ.shape
    d = D.shape[1]
    if ν_0 < d:
        ν_0 = d

    N_p = 100000
    N = np.prod(shape)

    θbar = θ.mean(axis=0)
    Dbar = D.mean(axis=0)

    Θ = (1 / (k - 1)) * np.einsum('ji,jk->ik', θ - θbar, θ - θbar)
    Ψ = (1 / (k - 1)) * np.einsum('ji,jk->ik', D - Dbar, θ - θbar)
    Δ = (1 / (k - 1)) * ((D - Dbar) ** 2).sum(axis=0)
    ν = k - 1 - n - 2
    invΘ = np.linalg.inv(Θ + Θ_0 / (k - 1))

    scale = [((k - 1) * (Δ[i] - np.einsum('i,ij,j->', Ψ[i, :], Θ, Ψ[i, :])) + C_0) for i in range(d)]

    C_ = np.array([invwishart(scale=scale[i], df=ν).rvs(size=shape[2]) for i in range(d)]).T
    C_ = C_[np.newaxis, np.newaxis, :]
    C_ = np.repeat(np.repeat(C_, shape[1], axis=1), shape[0], axis=0)

    L1 = np.sqrt(C_[0]) / np.sqrt(k - 1)
    L2 = np.linalg.cholesky(invΘ)
    L3 = np.sqrt(C_) / np.sqrt(k + 1)

    M_ = Ψ @ invΘ + np.einsum('...jk,...kl,ml->...jm', L1, np.random.randn(shape[1], shape[2], d, n), L2)
    M_ = M_[np.newaxis, :]
    M_ = np.repeat(M_, shape[0], axis=0)

    m_ = (k / (k + 1)) * (Dbar - M_ @ θbar) + np.einsum('...jk,...k->...j', L3, np.random.randn(*shape, d))

    model = LinearModel(m=m_, M=M_, C=C_, *args, **kwargs)

    samples = model.posterior(D_0).rvs(size=N_p // N).reshape((N_p // N) * N, n)

    logpdf = np.log(model.posterior(D_0).pdf(samples).mean(axis=(-1, -2, -3))).reshape((N_p // N) * N)

    return model, samples, logpdf


def LSBI_comp(θ, D, C_0=0, ν_0=0,  *args, **kwargs):
    '''
    Implementation of Linear Simulation-Based Inference (LSBI).
    Here, n denotes the number of parameters, d is the dimension of the dataset, and k is the number of simulated parameter-data pairs

    Parameters
    ----------
    θ : array-like, dimensions (k,n)
        Vector of parameter samples for each simulator run
    D : array-like, dimensions (k,d)
        Vector of data samples for each simulator run
    shape : int or tuple
        Number of samples drawn from posterior distributions
        Can be made a tuple to get an array of custom dimensions
    C_0 : array-like, dimensions (d,d)
        Prior for the covariance matrix C; default = 0
    ν_0 : int
        Prior degrees of freedom; default = 0
    Θ_0 : array-like, dimensions (n,n)
        Prior for model matrix M (can be different to the prior covariance), defualt = 0

    Returns
    -------
    LinearModel object with updated likelihood hyper-parameters
    '''
    full_model = LSBI(θ, D, shape=1)
    C_ = full_model.C
    M_ = full_model.M
    m_ = full_model.m
    x = np.einsum('...ji,...jk,...k->...i', M_, np.linalg.inv(C_), (D - m_))
    model = LSBI(θ, x, *args, **kwargs)
    return full_model, model