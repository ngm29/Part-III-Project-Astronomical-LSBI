from lsbi.model import LinearModel
from scipy.stats import invwishart, multivariate_normal, matrix_normal
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from functions.multivariate_laplace import multivariate_laplace
from functions.simulations import lsbi


# settings -------------------

d, n, N = 50, 4, 10000

k_set = [106,200,1000,5000]
#k_set = np.floor(np.logspace(2.03,4,100)).astype('int32')

seed = 1000

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

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

# Hyperparameters of the prior θ = μ ± √ Σ
mu = np.zeros(n)
Sigma = np.eye(n)**2

for k in k_set:

    theta_0, D_0, samps, samps2, d_kl = lsbi(d, n, k, N, t=n,  m=m, M=M, C=C,
                                             mu=mu, Sigma=Sigma, simulator='uniform', seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


names = ["θ%s"%i for i in range(n)]
labels = ["θ_%s"%i for i in range(n)]
markers= {'θ0':theta_0[0],'θ1':theta_0[1],'θ2':theta_0[2],'θ3':theta_0[3]}
samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
samplesN = []
if plot_prior:
    samplesN.append(samples)
for i in range(len(k_set)):
    samplesN.append(MCSamples(samples=samples_set[i],names = names, labels = labels, label='%s Simulations'%k_set[i]))


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
    #plt.suptitle(r'Uniform likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../figures/uniform.pdf')
    plt.show()

if plot_dkl:
    plt.semilogx(k_set, D_KL)
    plt.xlabel('Number of Simulations')
    plt.ylabel('KL Divergence')
    #plt.title(r'Uniform likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../figures/uniform_dkl.pdf')
    plt.show()