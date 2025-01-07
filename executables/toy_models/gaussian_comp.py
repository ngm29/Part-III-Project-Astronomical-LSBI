import numpy as np
import timeit as t
import matplotlib.pyplot as plt
from scipy.stats import invwishart
from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal, dkl
from utils.LSBI import LSBI, LSBI_comp
from utils.toy_models import Simulator
from getdist import plots, MCSamples
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import utils as dyfunc

# choose simulator
simulator = 'gaussian'

# number of data points (d) and parameters (n)
d, n = 50, 4

#  number of simulations (k)
k = 50*d

# number of posterior samples (N)
N = 10000

# approx. standard deviation for covariance matrix
σ = 0.5

# degrees of freedom
ν = d+2

# random seed
seed = 1000
np.random.seed(seed)

plot_prior = 1      # plot prior?
plot_posterior = 1  # plot posterior?
plot_dkl = 0        # plot KL divergence?
plot_nested = 1    # plot nested sampling?

# Hyperparameters of the prior θ = μ ± √ Σ
μ = np.zeros(n)
Σ = np.eye(n)**2

# Hyperparameters of the likelihood D = m + M θ ± √ C
M = (1/d)*np.random.randn(d, n)
m = np.random.randn(d)
C = invwishart(df=ν, scale=(σ)**2*np.eye(d)).rvs()

# define observed data
np.random.seed(seed)
θ_0 = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=1)
D_0 = np.squeeze(Simulator(simulator,m,M,C,ν).rvs(θ_0))

# samples for prior
samples_0 = LinearModel(mu=μ, Sigma=Σ).prior().rvs(N)

# posterior samples for each k in k_set
samples = []

# KL Divergence for each k in k_set
D_KL = []

# without compression

print('Running %s simulations...' % k)
θ = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=k)
D = Simulator(simulator, m, M, C, ν).rvs(θ)

print('Estimating posterior...')
model = LSBI(θ, D, shape=N)
D_KL.append(np.mean(model.dkl(D_0)))
samples_1 = model.posterior(D_0).rvs()


# with compression

print('Running %s simulations...' % k)
θ = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=k)
D = Simulator(simulator, m, M, C, ν).rvs(θ)

print('Estimating posterior...')
full_model, model = LSBI_comp(θ, D, shape=N)
x_0 = np.einsum('...ji,...jk,...k->...i', full_model.M, np.linalg.inv(full_model.C), (D_0 - full_model.m))
D_KL.append(np.mean(model.dkl(x_0)))
samples_2 = model.posterior(x_0).rvs()

θ_0 = np.squeeze(θ_0)


# nested sampling estimation
if plot_nested:
    sampler = NestedSampler(lambda θ: Simulator(simulator,m,M,C,ν).logpdf(θ,D_0), lambda x: LinearModel(mu=μ, Sigma=Σ).prior().bijector(x), len(θ_0))
    sampler.run_nested()
    results = sampler.results
    samples_ns = results.samples  # samples
    weights_ns = results.importance_weights()

    μ_ns, Σ_ns = dyfunc.mean_and_cov(samples_ns, weights_ns)

    D_KL_ns = dkl(multivariate_normal(mean=μ_ns,cov=Σ_ns),multivariate_normal(mean=μ,cov=Σ))

# plotting

names = ["θ%s"%i for i in range(n)]
labels = ["θ_%s"%i for i in range(n)]
markers= {'θ0':θ_0[0],'θ1':θ_0[1],'θ2':θ_0[2],'θ3':θ_0[3]}

MCsamples = []
if plot_prior:
    MCsamples.append(MCSamples(samples=samples_0, names=names, labels=labels, label='Prior'))

MCsamples.append(MCSamples(samples=samples_1, names=names, labels=labels, label='Uncompressed'))

MCsamples.append(MCSamples(samples=samples_2, names=names, labels=labels, label='Compressed'))

# nested sampling
if plot_nested:
    MCsamples.append(MCSamples(samples=samples_ns, weights=weights_ns, names=names, labels=labels, label='Nested Sampling'))


if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    if plot_prior:
        g.triangle_plot([MCsamples[i] for i in range(len(MCsamples))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['#006FED', '#E03424', 'gray', '#009966', '#000866', '#336600', '#006633',
                                        'm', 'r'])
    else:
        g.triangle_plot([MCsamples[i] for i in range(len(MCsamples))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['#E03424', 'gray', '#009966', '#000866', '#336600', '#006633', 'm','r'])
    plt.suptitle(simulator + r' likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../../figures/toy_models/' + simulator + '_comp.pdf')
    plt.show()

