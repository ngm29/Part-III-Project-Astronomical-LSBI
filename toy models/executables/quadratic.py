from lsbi.model import LinearModel
from scipy.stats import invwishart, multivariate_normal, matrix_normal
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from functions.multivariate_laplace import multivariate_laplace
from functions.simulations import lsbi


# settings -------------------

d, n, k, N = 40, 4, 1000, 10000

seed = 1000

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# size of quadratic term
eps = 1

# ----------------------------

# posterior samples
samples_set = []

# KL Divergence
D_KL = []

np.random.seed(seed)

# Hyperparameters of the likelihood D = m + M θ ± √ C
M = (1/d)*np.random.randn(d, n)
m = np.random.randn(d)
C = invwishart(df=d+2, scale=(0.5)**2*np.eye(d)).rvs()

Q = eps*(1/d)*np.random.randn(n,d, n)

# Hyperparameters of the prior θ = μ ± √ Σ
mu = np.zeros(n)
Sigma = np.eye(n)**2

model = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)





np.random.seed(seed)
theta_0, D_0, samps, samps2, d_kl = lsbi(d, n, k, N, t=n,  m=m, M=M, C=C,
                                             mu=mu, Sigma=Sigma, simulator='Gaussian', seed=seed)





names = ["θ%s"%i for i in range(n)]
labels = ["θ_%s"%i for i in range(n)]
markers= {'θ0':theta_0[0],'θ1':theta_0[1],'θ2':theta_0[2],'θ3':theta_0[3]}
samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
samplesN = []
if plot_prior:
    samplesN.append(samples)

samplesN.append(MCSamples(samples=samps2,names = names, labels = labels, label=r'$\epsilon = %s$'%eps))

# true posterior
model = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)
samps1 = model.posterior(D_0).rvs(size=N)
samplesN.append(MCSamples(samples=samps1,names = names, labels = labels, label='%s Simulations'%k))



if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    if plot_prior:
        g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm',
                                        'r'])
    else:
        g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm', 'r'])
    #plt.suptitle(r'Laplacian likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../figures/quadratic.pdf')
    plt.show()

