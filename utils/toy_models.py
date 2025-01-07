import numpy as np
from numpy.linalg import det, slogdet, inv, norm
from scipy.odr import quadratic
from scipy.stats import  uniform, chi2, expon
from scipy.special import gamma, kn
from lsbi.stats import multivariate_normal
from utils.multivariate_laplace import multivariate_laplace


class Simulator(object):
    def __init__(self, simulator, m, M, C, ν=0, Q=0):
        self.simulator = simulator
        self.m = m
        self.M = M
        self.C = C
        self.ν = ν
        self.Q = Q
        self.quadratic = quadratic

    def model(self, θ):
        return self.m + np.einsum("...ij,...j->...i", self.M, θ)

    def model_quad(self, θ):
        return  self.model(θ) + np.einsum("...i,...ijk,...k->...j",θ, self.Q, θ)

    def rvs(self, θ):
        if norm(self.Q) == 0:
            model = self.model
        else:
            model = self.model_quad

        if self.simulator == 'gaussian':
            return multivariate_normal(model(θ), self.C).rvs()
        elif self.simulator == 'uniform':
            scale = np.sqrt(np.abs(np.diagonal(self.C)))
            loc = model(θ) - scale/2
            return uniform.rvs(loc, scale)
        elif self.simulator == 'student-t':
            k = θ.shape[0]
            u = chi2.rvs(df=self.ν, size=k)
            y = multivariate_normal(0, self.C).rvs(size=k)
            return model(θ) + np.einsum('...,...i->...i', np.sqrt(self.ν / u), y)
        elif self.simulator == 'laplacian':
            return multivariate_laplace.rvs(model(θ), self.C)
        else:
            raise ValueError('Unrecognised toy model')

    def logpdf(self, θ, x):
        if norm(self.Q == 0) :
            model = self.model
        else:
            model = self.model_quad

        if self.simulator == 'gaussian':
            return multivariate_normal(model(θ), self.C).logpdf(x)
        if self.simulator == 'uniform':
            scale = np.sqrt(np.abs(np.diagonal(self.C)))
            loc = model(θ) - scale / 2
            return uniform.logpdf(x,loc, scale)
        elif self.simulator == 'student-t':
            d = x.shape[-1]
            ν = self.ν
            lognorm = np.log(gamma((ν + d) / 2)) - np.log(gamma(ν / 2) * (ν * np.pi) ** (d / 2)) - 0.5 * \
                      slogdet(self.C)[1]
            return lognorm - (ν + d) / 2 * np.log(
                1 + (1 / ν) * np.einsum('...i,...ij,...j->...', (x - model(θ)), inv(self.C), (x - model(θ))))
        elif self.simulator == 'laplacian':
            return multivariate_laplace.logpdf(x,mean=model(θ), cov=self.C)
        else:
            raise ValueError('Unrecognised toy model')