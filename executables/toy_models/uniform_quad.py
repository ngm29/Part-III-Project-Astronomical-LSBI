import numpy as np
import timeit as t
import matplotlib.pyplot as plt
from scipy.stats import invwishart, uniform
from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal, dkl
from utils.LSBI import LSBI
from utils.toy_models import Simulator
from getdist import plots, MCSamples
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import utils as dyfunc

# choose simulator
simulator = 'uniform'
quadratic = True

# number of data points (d) and parameters (n)
d, n = 50, 4

# number of LSBI rounds
rounds = 5

# (set of) number of simulations (k)
k = 2500                                   # for corner plot
#k_set = np.floor(np.logspace(2.03,4,100)).astype('int32')      # for d_kl plot

# number of posterior samples (N)
N = 1000

# approx. standard deviation for covariance matrix
σ = 0.5

# degrees of freedom
ν = d+2

# size of quadratic term
eps_set = [0.01,0.1,1]

# random seed
seed = 1
np.random.seed(seed)

plot_prior = 1      # plot prior?
plot_posterior = 1  # plot posterior?
plot_dkl = 1        # plot KL divergence?
plot_nested = 1     # plot nested sampling?

# Hyperparameters of the prior θ = μ ± √ Σ
μ = np.zeros(n)
Σ = np.eye(n)**2

# Hyperparameters of the likelihood D = m + M θ ± √ C
M = (1/d)*np.random.randn(d, n)
Q = (1/d)*np.random.randn(n, d, n)
m = np.random.randn(d)
C = invwishart(df=ν, scale=(σ)**2*np.eye(d)).rvs()

# define observed data
np.random.seed(seed)
θ_0 = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=1)
D_0 = np.squeeze(Simulator(simulator, m, M, C, ν, Q).rvs(θ_0))

# posterior samples for each k in k_set
Samples = []

# logpdf if the samples for D_KL computation
LogPDF = []

# samples for prior
samples_0 = LinearModel(mu=μ, Sigma=Σ).prior().rvs(100000)
Samples.append(samples_0)


# simulations
for r in range(rounds):
    print(f'Round {r + 1} of {rounds}')

    if not r:
        print('Running %s simulations...' % k)
        θ = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=k)

    else:
        print('Running %s simulations...' % k)
        posterior = model.posterior(D_0)
        θ = posterior.rvs(size=k//N).reshape((k//N)*N, n)
    D = Simulator(simulator, m, M, C, ν, Q).rvs(θ)

    print('Estimating posterior...')

    model, samples, logpdf = LSBI(θ, D, D_0, μ=μ, Σ=Σ, shape=N)
    logpdf_0 = LinearModel(μ=μ, Σ=Σ).prior().logpdf(samples, broadcast=True)
    Samples.append(samples)
    LogPDF.append((logpdf,logpdf_0))

θ_0 = np.squeeze(θ_0)


size = 100

D_KL_ABC = []

for j in range(size):

    # Rejection ABC

    k_ABC = 1000000
    θ = LinearModel(mu=μ, Sigma=Σ).prior().rvs(size=k_ABC)
    D = Simulator(simulator, m, M, C, ν,Q).rvs(θ)

    scale = np.sqrt(np.abs(np.diagonal(C)))
    loc = m + np.einsum("...ij,...j->...i", M, θ_0)  + np.einsum("...i,...ijk,...k->...j",θ_0, Q, θ_0)

    samples_ABC = []

    for i in range(len(θ)):
        if np.all(uniform.pdf(D[i], loc=loc-scale/2,scale=scale)):
            samples_ABC.append(θ[i])

    samples_ABC = np.array(samples_ABC)

    print(np.shape(samples_ABC))


    μ_ABC = samples_ABC.mean(axis=0)
    Σ_ABC = np.cov(samples_ABC.T)

    D_KL_ABC.append(dkl(LinearModel(mu=μ_ABC, Sigma=Σ_ABC).prior(),LinearModel(mu=μ, Sigma=Σ).prior()))


D_KL_avg = np.mean(D_KL_ABC)+0.5
D_KL_std = np.std(D_KL_ABC)



# KL divergence
D_KL = [np.mean(LogPDF[i][0] - LogPDF[i][1]) for i in range(len(LogPDF))]


# plotting

names = ["θ%s"%i for i in range(n)]
labels = ["θ_%s"%i for i in range(n)]
markers= {'θ0':θ_0[0],'θ1':θ_0[1],'θ2':θ_0[2],'θ3':θ_0[3]}

MCsamples = []
for r in range(rounds + 1):
    if not r:
        MCsamples.append(MCSamples(samples=Samples[r], names=names, labels=labels, label='Prior'))
    else:
        MCsamples.append(MCSamples(samples=Samples[r], names=names, labels=labels, label='Round %s' % r))

# Rejection ABC
if plot_nested:
    MCsamples.append(MCSamples(samples=samples_ABC, names=names, labels=labels, label='Rejection ABC'))

# true posterior
if simulator=='gaussian' and not quadratic:
    model = LinearModel(M=M, m=m, C=C, mu=μ, Sigma=Σ)
    samples_1 = model.posterior(D_0).rvs(size=N)
    MCsamples.append(MCSamples(samples=samples_1, names=names, labels=labels, label='True Posterior'))

if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    if plot_prior:
        g.triangle_plot([MCsamples[i] for i in range(len(MCsamples))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['gray', '#006FED', '#E03424','green', 'orangered', '#000866', '#336600',
                                        'm', 'r'])
    else:
        g.triangle_plot([MCsamples[i] for i in range(len(MCsamples))], markers=markers, filled=True,
                        legend_loc='upper right',
                        contour_colors=['#006FED', '#E03424','green', 'orangered', '#000866', '#336600',
                                        'm', 'r'])
    #plt.suptitle(simulator + r' likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../../figures/toy_models/'+simulator+'_quad.pdf')
    plt.show()

if plot_dkl:
    if plot_nested:
        eps = D_KL_std
        plt.axhline(y=D_KL_avg, color='black', linewidth=.75)
        plt.fill_between(np.arange(1, r + 1), D_KL_avg - eps, D_KL_avg + eps, color='lightgray')
    # plt.axhline(y=D_KL_ns+eps, color='gray', linestyle='--')
    # plt.axhline(y=D_KL_ns-eps, color='gray', linestyle='--')
    plt.plot(np.arange(1, r + 1), D_KL)
    plt.margins(x=0)
    plt.xlabel('Number of Rounds')
    plt.ylabel('KL Divergence')
    # plt.title(simulator + r' likelihood, $n=%s$, $d=%s$'%(n,d))
    plt.savefig('../../figures/toy_models/' + simulator + '_quad_dkl.pdf')
    plt.show()