from lsbi.model import LinearModel
from scipy.stats import invwishart,  multivariate_normal, matrix_normal
import numpy as np
from globalemu.eval import evaluate
import timeit




def lsbi(d, n, k, N, c, batch=1, comp=1, m=0, M=0, C=1, mu=0, Sigma=1,
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

    if d % comp:
        d_comp = d // comp +1
    else:
        d_comp = d // comp

    D_0 = np.array(simulator(theta_0, comp))

    # generate data points
    D = np.zeros((k, d_comp))
    print('Running %s simulations...' % k)
    for i in range(k):
        if i % (k // 10) == 0:
            start = timeit.default_timer()
        if i % (k // 10) == 1:
            print((i - 1) / k * 100, '% done. Time remaining: approx.',
                  np.floor((timeit.default_timer() - start) * (k - i + 1)), 'seconds.')
        signal = simulator(theta[i, :], comp)
        D[i] = np.array(signal)

    # means
    theta_mean = (1 / k) * np.sum(theta, axis=0)
    D_mean = (1 / k) * np.sum(D, axis=0)

    # covariances
    theta_cov = (1 / k) * np.einsum('ji,jk->ik', theta - theta_mean, theta - theta_mean)
    D_cov = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, D - D_mean)
    corr = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, theta - theta_mean)

    # utilities
    nu = k - d_comp - n - 2
    inv_theta_cov = np.linalg.inv(theta_cov)
    scale = k * (D_cov - np.einsum('ij,jk,lk->il', corr, inv_theta_cov, corr))
    loc = np.einsum('ij,jk->ik', corr, inv_theta_cov)

    if svd:
        U, S, Vh = np.linalg.svd(loc)
        s = S
        S = np.eye(d_comp, n) * S
        loc = np.einsum('ij,jk->ik', U[:, :c], S[:c, :c])
        theta_mean = np.einsum('ij,j->i', Vh[:c, :], theta_mean)
        theta_cov = np.einsum('ij,jk,lk->il', Vh[:c, :], theta_cov, Vh[:c, :])
        mu = np.einsum('ij,j->i', Vh[:c, :], mu)
        Sigma = np.einsum('ij,jk,lk->il', Vh[:c, :], Sigma, Vh[:c, :])
        theta_0 = np.einsum('ij,j->i', Vh[:c, :], theta_0)
    else:
        c = n


    inv_theta_cov = np.linalg.inv(theta_cov)

    # estimate posterior

    print('Estimating posterior...')
    start = timeit.default_timer()
    # draw from posteriors of m, M, and C
    C_temp = invwishart.rvs(df=nu, scale=scale, size=N)

    M_temp = matrix_normal.rvs(np.zeros((d_comp, c)), np.eye(d_comp), np.eye(c), size=N)
    A = np.linalg.cholesky((1 / k) * C_temp)
    B = np.linalg.cholesky(inv_theta_cov)
    M_temp = loc + np.einsum('...ij,...jk,lk->...il', A, M_temp, B)
    m_temp = multivariate_normal.rvs(np.zeros((d_comp,)), np.eye(d_comp), size=N)
    m_temp = (D_mean - np.einsum('...ij,j->...i', M_temp, theta_mean)
              + np.einsum('...ij,...j->...i', A, m_temp))

    # calculate posterior and dkl
    model_temp = LinearModel(M=M_temp, m=m_temp, C=C_temp, mu=mu, Sigma=Sigma)
    d_kl = model_temp.dkl(D_0)
    samps = model_temp.posterior(D_0).rvs()

    samps0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=N)

    print(timeit.default_timer()-start)
    return theta_0, D_0, theta, D, m_temp, M_temp, C_temp, samps0, samps, d_kl







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

    if d % comp:
        d_comp = d // comp + 1
    else:
        d_comp = d // comp

    D_0 = np.array(simulator(theta_0, comp))

    # generate data points
    D = np.zeros((k, d_comp))
    print('Running %s simulations...' % k)
    for i in range(k):
        if i % (k // 10) == 0:
            start = timeit.default_timer()
        if i % (k // 10) == 1:
            print((i - 1) / k * 100, '% done. Time remaining: approx.',
                  np.floor((timeit.default_timer() - start) * (k - i + 1)), 'seconds.')
        signal = simulator(theta[i, :], comp)
        D[i] = np.array(signal)

    # means
    theta_mean = (1 / k) * np.sum(theta, axis=0)
    D_mean = (1 / k) * np.sum(D, axis=0)

    # covariances
    theta_cov = (1 / k) * np.einsum('ji,jk->ik', theta - theta_mean, theta - theta_mean)
    D_cov = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, D - D_mean)
    corr = (1 / k) * np.einsum('ji,jk->ik', D - D_mean, theta - theta_mean)

    # utilities
    nu = k - d_comp - n - 2
    inv_theta_cov = np.linalg.inv(theta_cov)
    scale = k * (D_cov - np.einsum('ij,jk,lk->il', corr, inv_theta_cov, corr))
    loc = np.einsum('ij,jk->ik', corr, inv_theta_cov)
    rescale = np.linalg.inv(np.einsum('ji,jk,kl->il',loc,np.linalg.inv(scale),loc))



    C_temp = invwishart.rvs(df=nu, scale=scale)

    M_temp = matrix_normal.rvs(np.zeros((d_comp, n)), np.eye(d_comp), np.eye(n))
    A = np.linalg.cholesky((1 / k) * C_temp)
    B = np.linalg.cholesky(inv_theta_cov)
    M_temp = loc + np.einsum('...ij,...jk,lk->...il', A, M_temp, B)
    m_temp = multivariate_normal.rvs(np.zeros((d_comp,)), np.eye(d_comp))
    m_temp = (D_mean - np.einsum('...ij,j->...i', M_temp, theta_mean)
              + np.einsum('...ij,...j->...i', A, m_temp))

    Gamma = np.linalg.inv(np.einsum('ji,jk,kl->il',M_temp,np.linalg.inv(C_temp),M_temp))

    x_0 = np.einsum('ij,kj,kl,l->i',Gamma,M_temp,np.linalg.inv(C_temp),(D_0-m_temp))

    # estimate posterior
    print('Estimating posterior...')
    G_temp = invwishart.rvs(df=nu, scale=rescale, size = N)


    # calculate posterior and dkl
    model_temp = LinearModel( C=G_temp, mu=mu, Sigma=Sigma)
    d_kl = model_temp.dkl(x_0)
    samps = model_temp.posterior(x_0).rvs()

    samps0 = LinearModel(mu=mu, Sigma=Sigma).prior().rvs(size=N)

    return theta_0, D_0, x_0, samps0, samps, d_kl

