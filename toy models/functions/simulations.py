from lsbi.model import LinearModel
from scipy.stats import invwishart, multivariate_t, multivariate_normal, matrix_normal, uniform
import numpy as np
from functions.multivariate_laplace import multivariate_laplace
import timeit


def lsbi(d, n, k, N, t, comp=1, m=0, M=0, Q=0, C=1, mu=0, Sigma=1,
         theta=[], theta_0 = [], simulator='None', seed =None, svd=False):

    # set random seed
    np.random.seed(seed)

    # draw random samples of theta
    if not len(theta):
        theta = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=k)

    # define observed data
    if not len(theta_0):
        np.random.seed(seed)
        theta_0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs()

    # generate data points
    D = np.zeros((k,d))
    print('Running %s simulations...'%k)
    if simulator == 'Gaussian':
        model = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)
        np.random.seed(seed)
        D_0 = model.likelihood(theta_0).rvs()
        for i in range(k):
            D[i] = model.likelihood(theta[i]).rvs()
    elif simulator == 'Laplacian':
        loc = m + np.einsum('ij,j->i', M, theta_0)
        np.random.seed(seed)
        D_0 = multivariate_laplace.rvs(loc, C)[0, :]
        for i in range(k):
            loc = m + np.einsum('ij,j->i', M, theta[i])
            D[i] = multivariate_laplace.rvs(loc, C)[0, :]
    elif simulator == 'uniform':
        loc = m + np.einsum('ij,j->i', M, theta_0)
        scale = np.sqrt(np.abs(np.diagonal(C)))
        loc = loc - scale / 2
        np.random.seed(seed)
        D_0 = uniform.rvs(loc,scale)
        for i in range(k):
            loc = m + np.einsum('ij,j->i', M, theta[i])
            scale = np.sqrt(np.abs(np.diagonal(C)))
            loc = loc - scale / 2
            D[i] = uniform.rvs(loc,scale)
    elif simulator == 'Student-t':
        loc = m + np.einsum('ij,j->i', M, theta_0)
        np.random.seed(seed)
        D_0 = np.atleast_1d(multivariate_t.rvs(loc, C))
        for i in range(k):
            loc = m + np.einsum('ij,j->i', M, theta[i])
            D[i] = np.atleast_1d(multivariate_t.rvs(loc, C))
    elif simulator == 'quadratic':
        np.random.seed(seed)
        loc = m + np.einsum('ij,j->i', M, theta_0) + np.einsum('i,ijk,k->j', theta_0, Q, theta_0)
        L = np.linalg.cholesky(C)
        D_0 = multivariate_normal.rvs(np.zeros((d,)),np.eye(d))
        D_0 = loc + np.einsum('ij,j->i',L,D_0)
        for i in range(k):
            loc = m + np.einsum('ij,j->i', M, theta[i])  + np.einsum('i,ijk,k->j', theta_0, Q, theta_0)
            L = np.linalg.cholesky(C)
            D[i] = multivariate_normal.rvs(np.zeros((d,)), np.eye(d))
            D[i] = loc + np.einsum('ij,j->i', L, D[i])
    else:
        print('Please choose a model: Gaussian, Laplacian, Student-t, uniform')
        return None


    # means
    theta_mean = (1 / k) * np.sum(theta, axis=0)
    D_mean = (1 / k) * np.sum(D, axis=0)

    # covariances
    theta_cov = (1 / k) * np.einsum('ji,jk->ik', theta - theta_mean, theta - theta_mean)
    D_cov = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, D - D_mean)
    corr = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, theta - theta_mean)

    # utilities
    nu = k - d - n - 2
    inv_theta_cov = np.linalg.inv(theta_cov)
    scale = k * (D_cov - np.einsum('ij,jk,lk->il', corr, inv_theta_cov, corr))
    loc = np.einsum('ij,jk->ik', corr, inv_theta_cov)

    # optionally perform svd truncation
    if svd:
        U, S, Vh = np.linalg.svd(loc)
        S = np.eye(d, n) * S
        loc = np.einsum('ij,jk->ik', U[:, :t], S[:t, :t])
        theta_mean = np.einsum('ij,j->i', Vh[:t, :], theta_mean)
        theta_cov = np.einsum('ij,jk,lk->il', Vh[:t, :], theta_cov, Vh[:t, :])
        mu = np.einsum('ij,j->i', Vh[:t, :], mu)
        Sigma = np.einsum('ij,jk,lk->il', Vh[:t, :], Sigma, Vh[:t, :])
        theta_0 = np.einsum('ij,j->i', Vh[:t, :], theta_0)
    else:
        t = n

    inv_theta_cov = np.linalg.inv(theta_cov)

    print('Estimating posterior...')

    # draw from posteriors of m, M, and C
    C_temp = invwishart.rvs(df=nu, scale=scale, size=N)

    M_temp = matrix_normal.rvs(np.zeros((d,t)),np.eye(d),np.eye(t),size=N)
    A = np.linalg.cholesky((1 / k) * C_temp)
    B = np.linalg.cholesky(inv_theta_cov)
    M_temp = loc + np.einsum('...ij,...jk,lk->...il', A, M_temp, B)
    m_temp = multivariate_normal.rvs(np.zeros((d,)),np.eye(d),size=N)
    m_temp = (D_mean - np.einsum('...ij,j->...i', M_temp, theta_mean)
              + np.einsum('...ij,...j->...i',A,m_temp))

    # calculate posterior and dkl
    model_temp = LinearModel(M=M_temp, m=m_temp, C=C_temp, mu=mu, Sigma=Sigma)
    d_kl = model_temp.dkl(D_0)
    samps = model_temp.posterior(D_0).rvs()

    samps0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=N)
    return theta_0, D_0, samps0, samps, d_kl













def lsbi_comp(d, n, k, N, batch=1, comp=1, m=0, M=0, C=1, mu=0, Sigma=1,
         theta=[], theta_0 = [], simulator='None', seed =None):

    # set random seed
    np.random.seed(seed)

    # draw random samples of theta
    if not len(theta):
        theta = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=k)

    # define observed data
    if not len(theta_0):
        np.random.seed(seed)
        theta_0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs()

    # generate data points
    D = np.zeros((k,d))
    print('Running %s simulations...'%k)
    if simulator == 'Gaussian':
        model = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)
        np.random.seed(seed)
        D_0 = model.likelihood(theta_0).rvs()
        for i in range(k):
            if i % (k // 10) == 0:
                print(i / k * 100, '% done.')
            D[i] = model.likelihood(theta[i]).rvs()
    elif simulator == 'Laplacian':
        loc = m + np.einsum('ij,j->i', M, theta_0)
        np.random.seed(seed)
        D_0 = multivariate_laplace(loc, C).rvs()[0, :]
        for i in range(k):
            if i % (k // 10) == 0:
                print(i / k * 100, '% done.')
            loc = m + np.einsum('ij,j->i', M, theta[i])
            D[i] = multivariate_laplace(loc, C).rvs()[0, :]
    elif simulator == 'Student-t':
        loc = m + np.einsum('ij,j->i', M, theta_0)
        np.random.seed(seed)
        D_0 = np.atleast_1d(multivariate_t(loc, C).rvs())
        for i in range(k):
            if i % (k // 10) == 0:
                print(i / k * 100, '% done.')
            loc = m + np.einsum('ij,j->i', M, theta[i])
            D[i] = np.atleast_1d(multivariate_t(loc, C).rvs())
    else:
        print('Please choose a model: Gaussian, Laplacian, Student-t')
        return None

    # means
    theta_mean = (1 / k) * np.sum(theta, axis=0)
    D_mean = (1 / k) * np.sum(D, axis=0)

    # covariances
    theta_cov = (1 / k) * np.einsum('ji,jk->ik', theta - theta_mean, theta - theta_mean)
    D_cov = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, D - D_mean)
    corr = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, theta - theta_mean)

    # utilities
    nu = k - d - n - 2
    inv_theta_cov = np.linalg.inv(theta_cov)
    scale = k * (D_cov - np.einsum('ij,jk,lk->il', corr, inv_theta_cov, corr))
    loc = np.einsum('ij,jk->ik', corr, inv_theta_cov)
    rescale = np.linalg.inv(np.einsum('ji,jk,kl->il',loc,np.linalg.inv(scale),loc))

    C_temp = invwishart(df=nu, scale=scale).rvs()
    M_temp = multivariate_normal().rvs(size=(d, n))
    A = np.linalg.cholesky((1 / k) * C_temp)
    B = np.linalg.cholesky(inv_theta_cov)
    M_temp = loc + np.einsum('ij,jk,lk->il', A, M_temp, B)
    m_temp = multivariate_normal(D_mean - np.einsum('ij,j->i', M_temp, theta_mean),
                                 (1 / k) * C_temp, allow_singular=True).rvs()
    Gamma = np.linalg.inv(np.einsum('ji,jk,kl->il',M_temp,np.linalg.inv(C_temp),M_temp))



    x_0 = np.einsum('ij,kj,kl,l->i',Gamma,M_temp,np.linalg.inv(C_temp),(D_0-m_temp))

    # estimate posterior

    N = N // batch
    samps = np.zeros((N * batch, n))
    d_kl = np.zeros(N)
    print('Estimating posterior...')
    for i in range(N):
        if i % (N // 10) == 0:
            start = timeit.default_timer()
        if i % (N // 10) == 1:
            print((i - 1) / N * 100, '% done. Time remaining: approx.',
                  np.floor((timeit.default_timer() - start) * (N - i + 1)), 'seconds.')
        G_temp = invwishart(df=nu, scale=rescale).rvs()
        model_temp = LinearModel( C=G_temp, mu=mu, Sigma=Sigma)
        d_kl[i] = model_temp.dkl(x_0)
        for j in range(batch):
            sample = model_temp.posterior(x_0).rvs()
            samps[(i * batch + j)] = sample

    samps0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=N * batch)
    return theta_0, D_0, x_0, samps0, samps, d_kl