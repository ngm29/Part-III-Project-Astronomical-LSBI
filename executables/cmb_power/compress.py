import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import invwishart, chi2
from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal, dkl
from utils.LSBI import LSBI_comp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
from getdist import plots, MCSamples
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import utils as dyfunc


# number of LSBI rounds
rounds = 5

# number of simulations per round
k = 10000

# number of posterior samples
N = 10

# random seed
seed = 0
np.random.seed(seed)

# emulator
emulator = CosmoPowerJAX(probe='cmb_tt')

l = np.arange(2, 2509)

# adding noise
class CMB(object):
    def __init__(self, Cl):
        self.Cl = Cl

    def rvs(self, shape=()):
        shape = tuple(np.atleast_1d(shape))
        return chi2(2*l+1).rvs(shape + self.Cl.shape)*self.Cl/(2*l+1)

    def logpdf(self, x):
        return (chi2(2*l+1).logpdf((2*l+1)*x/self.Cl)  + np.log(2*l+1)-np.log(self.Cl)).sum(axis=-1)


# prior
θmin, θmax = np.array([[0.01865, 0.02625], [0.05, 0.255], [0.64, 0.82], [0.04, 0.12], [0.84, 1.1], [1.61, 3.91]]).T

μ = (θmin + θmax)/2
Σ = (θmax - θmin)**2

# posterior samples for each k in k_set
samples = []

# KL Divergence for each k in k_set
D_KL = []

# define observed data
np.random.seed(seed)
θ_0 = np.random.uniform(θmin, θmax, size=(1,6))
D_0 = CMB(emulator.predict(θ_0)).rvs()
θ_0 = θ_0[0]

# samples for prior
model = LinearModel(μ=μ, Σ=Σ)
samples_0 = model.prior().rvs(size=100000)
samples.append(samples_0)

for r in range(rounds):

    print(f'Round {r + 1} of {rounds}')
    start = timeit.default_timer()

    if not r:
        print('Running %s simulations...' % k)
        θ = model.posterior(x_0).rvs(size=k // N).reshape(k, 6)
    else:
        print('Running %s simulations...' % k)
        θ = model.posterior(D_0).rvs(size=k // N).reshape(k, 6)
    D = CMB(emulator.predict(θ)).rvs()

    time = timeit.default_timer() - start
    print(f'Time elapsed: {round(time, 2)} seconds.')

    print('Estimating posterior...')
    start = timeit.default_timer()

    full_model, model = LSBI_comp(θ, D, μ=μ, Σ=Σ, shape=N)
    x_0 = np.einsum('...ji,...jk,...k->...i', full_model.M, np.linalg.inv(full_model.C), (D_0 - full_model.m))
    D_KL.append(np.mean(model.dkl(x_0)))
    samples.append(model.posterior(x_0).rvs(size=100000 // N).reshape(100000, 6))

    time = timeit.default_timer() - start
    print(f'Time elapsed: {round(time, 2)} seconds.')
    print(' ')

# nested sampling
paramnames = [('Ωbh2', r'\Omega_b h^2'), ('Ωch2', r'\Omega_c h^2'), ('h', 'h'), ('τ', r'\tau'), ('ns', r'n_s'), ('lnA', r'\ln(10^{10}A_s)')]
sampler = NestedSampler(lambda θ: CMB(emulator.predict(θ)).logpdf(D_0), lambda x: θmin + (θmax-θmin)*x, len(θmin))
sampler.run_nested()
results = sampler.results
samples_ns = results.samples  # samples
weights_ns = results.importance_weights()

μ_ns, Σ_ns = dyfunc.mean_and_cov(samples_ns, weights_ns)

D_KL_ns = dkl(multivariate_normal(mean=μ_ns,cov=Σ_ns),multivariate_normal(mean=μ,cov=Σ))

# plotting
names = ["θ%s" % i for i in range(6)]
labels = [r'\Omega_b h^2', r'\Omega_c h^2', 'h', r'\tau', r'\ln(10^{10}A_s)', r'n_s']
markers = {'θ0': θ_0[0], 'θ1': θ_0[1], 'θ2': θ_0[2], 'θ3': θ_0[3], 'θ4': θ_0[4], 'θ5': θ_0[5]}

MCsamples = []
for r in range(rounds + 1):
    if not r:
        MCsamples.append(MCSamples(samples=samples[r], names=names, labels=labels, label='Prior'))
    else:
        MCsamples.append(MCSamples(samples=samples[r], names=names, labels=labels, label='Round %s' % r))

MCsamples.append(MCSamples(samples=samples_ns, weights=weights_ns, names=names, labels=labels, label='Ground Truth'))

show_rounds_from = 3
R = show_rounds_from

g = plots.get_subplot_plotter()
g.settings.legend_fontsize = 15
g.settings.axes_labelsize = 15
g.triangle_plot([MCsamples[i + R] for i in range(len(MCsamples) - R)], markers=markers, filled=True,
                legend_loc='upper right',
                contour_colors=['gray', '#006FED', '#E03424', '#009966', '#000866', '#336600', 'm', 'r'])
plt.suptitle('Posterior estimate for CosmoPowerJAX, %s simulations per round' % k)
#plt.savefig('../../figures/CPJAX.pdf')
plt.show()

# KL Divergence
eps = 0.25
plt.axhline(y=D_KL_ns+eps, color='gray', linestyle='--')
plt.axhline(y=D_KL_ns-eps, color='gray', linestyle='--')
plt.plot(np.arange(1,r+1),D_KL)
plt.xlabel('Number of Rounds')
plt.ylabel('KL Divergence')
plt.title('KL Divergence for CosmoPowerJAX, %s simulations per round'%k)
#plt.savefig('../../figures/CPJAX_dkl.pdf')
plt.show()