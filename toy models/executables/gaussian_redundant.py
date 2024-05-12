from lsbi.model import LinearModel
from scipy.stats import invwishart, multivariate_normal, matrix_normal
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from functions.simulations import lsbi

# settings -------------------

d, n, N, batch = 10, 10, 1000, 10

k_set = [10000]

model_seed = 10000

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# how many relevant parameters
c = 4
# error for small singular values
eps_set = [0, 0.01, 0.1, 1]


# ----------------------------

# posterior samples
samples_set = []

for eps in eps_set:

    np.random.seed(model_seed)

    # Hyperparameters of the likelihood D = m + M θ ± √ C
    U = np.random.randn(d,d)
    V = np.random.randn(n,n)
    M = np.zeros((d,n))
    for i in range(n):
        M[i,i] = np.random.uniform(-eps,eps)
    for i in range(c):
        M[i,i] = np.random.uniform(-1,1)
    M = (1/d)*np.einsum('ij,jk,lk->il', U, M, V)
    m = np.random.randn(d)
    C = invwishart(df=d+2, scale=(0.5)**2*np.eye(d)).rvs()

    # Hyperparameters of the prior θ = μ ± √ Σ
    mu = np.zeros(n)
    Sigma = np.eye(n)**2

    for k in k_set:

        theta_0, D_0, samps, samps2, dkl = lsbi(d, n, k, N, c, m=m, M=M, C=C,
                                                 mu=mu, Sigma=Sigma, simulator='Gaussian', seed=model_seed, svd=True)
        samples_set.append(samps2)


    names = ["θ%s"%i for i in range(c)]
    labels = ["θ_%s"%i for i in range(c)]
    markers= {'θ0':theta_0[0],'θ1':theta_0[1]}
    samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
    samplesN = []
    if plot_prior:
        samplesN.append(samples)
    for i in range(len(k_set)):
        samplesN.append(MCSamples(samples=samples_set[i],names = names, labels = labels, label='%s Simulations'%k_set[i]))


    # true posterior
    U, S, Vh = np.linalg.svd(M)
    S = np.eye(d, n) * S
    M_new = np.einsum('ij,jk->ik', U[:, :c], S[:c, :c])
    mu_new = np.einsum('ij,j->i', Vh[:c, :], mu)
    Sigma_new = np.einsum('ij,jk,lk->il', Vh[:c, :], Sigma, Vh[:c, :])
    model = LinearModel(M=M_new, m=m, C=C, mu=mu_new, Sigma=Sigma_new)
    samps1 = model.posterior(D_0).rvs(size=N)
    samples1 = MCSamples(samples=samps1,names = names, labels = labels, label='True Posterior')
    samplesN.append(samples1)


    if plot_posterior:
        g = plots.get_subplot_plotter()
        g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True, legend_loc='upper right',
                            contour_colors=['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm', 'r'])
        #plt.suptitle(r'Gaussian likelihood, $n=%s$, $d=%s$'%(c,d))
        plt.show()
