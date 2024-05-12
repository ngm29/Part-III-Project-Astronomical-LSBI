from lsbi.model import LinearModel
from scipy.stats import invwishart, multivariate_normal, matrix_normal
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from functions.simulations import lsbi_comp, lsbi
import timeit

# settings -------------------

n, k, N =  4, 1000, 10000

d_set = [50]
#d_set = np.floor(np.logspace(1,2,20)).astype('int32')

seed = 1000

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# ----------------------------

# posterior samples
samples_set = []

# KL Divergence
D_KL = []

# times
times1 = []
times2 = []

for d in d_set:

    np.random.seed(seed)

    # Hyperparameters of the likelihood D = m + M θ ± √ C
    M = (1/d)*np.random.randn(d, n)
    m = np.random.randn(d)
    C = invwishart(df=d+2, scale=(0.5)**2*np.eye(d)).rvs()

    # Hyperparameters of the prior θ = μ ± √ Σ
    mu = np.zeros(n)
    Sigma = np.eye(n)**2

    start = timeit.default_timer()

    theta_0, D_0, x_0, samps, samps2, d_kl = lsbi_comp(d, n, k, N, m=m, M=M, C=C,
                                                     mu=mu, Sigma=Sigma, simulator='Gaussian', seed=seed)
    times1.append(timeit.default_timer()-start)

    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))

    start = timeit.default_timer()

    theta_0, D_0, samps, samps2, d_kl = lsbi(d, n, k, N, t=n, m=m, M=M, C=C,
                                                 mu=mu, Sigma=Sigma, simulator='Gaussian', seed=seed)

    times2.append(timeit.default_timer() - start)

    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


names = ["θ%s"%i for i in range(n)]
labels = ["θ_%s"%i for i in range(n)]
markers= {'θ0':theta_0[0],'θ1':theta_0[1],'θ2':theta_0[2],'θ3':theta_0[3]}
samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
samplesN = []
if plot_prior:
    samplesN.append(samples)
samplesN.append(MCSamples(samples=samples_set[1],names = names, labels = labels, label='Uncompressed'))
samplesN.append(MCSamples(samples=samples_set[0],names = names, labels = labels, label='Compressed'))



if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True, legend_loc='upper right',
                        contour_colors=['#006FED',  'gray', '#E03424', '#009966', '#000866', '#336600', '#006633', 'm', 'r'])
    #plt.suptitle(r'Gaussian likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../figures/gaussian_compress.pdf')
    plt.show()

plt.semilogy(d_set, times2, label='Uncompressed')
plt.semilogy(d_set, times1, label='Compressed')
plt.xlabel('Size of dataset')
plt.ylabel('Computational time (s)')
plt.legend()
#plt.title(r'Gaussian likelihood, $n=%s$, $d=%s$'%(n,d))
plt.savefig('../figures/gaussian_times.pdf')
plt.show()

