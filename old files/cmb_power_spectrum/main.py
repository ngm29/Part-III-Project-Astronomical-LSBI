import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from getdist import plots, MCSamples
from cmbemu.eval import evaluate
from lsbi.simulations import cmb_lsbi, cmb_mclsbi

### ------------- SETTINGS -------------------------- SETTINGS -------------------------- SETTINGS ------------- ###

# number of data points (d) and parameters (n)
d, n = 300, 6

# (set of) number of simulations (k)
#k_set = [608, 3000, 15000, 100000]
k_set = [608]               # for compress vs non-compress

# number of posterior samples (N)
N =  10000

# compression
comp = 1

# random seed
seed = 100
np.random.seed(seed)

plot_prior = 1      # plot prior?
plot_posterior = 1  # plot posterior?
plot_dkl = 0        # plot KL divergence?
save_figs = 0       # save figures?

### --------------- MODEL ------------------------------ MODEL ------------------------------ MODEL --------------- ###

# load data
data_dir = 'data/'
base_dir = 'results/'
params = np.loadtxt(data_dir + 'train_data.txt')
data = np.loadtxt(data_dir + 'train_labels.txt')

l = np.linspace(2, 2000, 300)

# KL Divergence
D_KL = []

# posterior samples
samples_set = []

# compute the prior
mu = np.mean(params, axis=0)
Sigma = np.eye(n)*np.std(params, axis=0) ** 2

### ------------ SIMULATIONS ------------------------ SIMULATIONS ------------------------ SIMULATIONS ------------ ###

# define simulator
predictor = evaluate(base_dir=base_dir)
def CMB_simulator(theta,comp):
    signal = predictor(theta)[0][::comp] * chi2.rvs(2 * l[::comp] + 1) / (2 * l[::comp] + 1)
    return signal

for k in k_set:

    theta_0, D_0, x_0, theta, D, samps, samps2, d_kl = cmb_mclsbi(CMB_simulator, d, k, N, mu, Sigma, comp=comp, seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


    theta_0, D_0, theta, D, m, M, C, samps, samps2, d_kl = cmb_lsbi(CMB_simulator, d, k, N, mu, Sigma, comp=comp, seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


### --------------- PLOTS ------------------------------ PLOTS ------------------------------ PLOTS --------------- ###

names = ["θ%s"%i for i in range(n)]
labels = [r'\Omega_b',r'\Omega_c',r'\tau',r'\ln(10^{10} A_s)',r'N_s',r'H_0']
markers= {'θ0':theta_0[0],'θ1':theta_0[1],'θ2':theta_0[2],'θ3':theta_0[3],'θ4':theta_0[4],'θ5':theta_0[5]}
samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
samplesN = []
if plot_prior:
    samplesN.append(samples)
#for i in range(len(k_set)):
#    samplesN.append(MCSamples(samples=samples_set[i],names = names, labels = labels, label='%s Simulations'%k_set[i]))

samplesN.append(MCSamples(samples=samples_set[1],names = names, labels = labels, label='Uncompressed'))
samplesN.append(MCSamples(samples=samples_set[0],names = names, labels = labels, label='Compressed'))


if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True, legend_loc='upper right',
                        contour_colors=['#006FED',  'gray', '#E03424', '#009966',  '#000866',  '#336600', 'm', 'r'])
    if save_figs:
        plt.savefig('figures/CMB_posterior.pdf')
    plt.show()

if plot_dkl:
    plt.semilogx(k_set, D_KL)
    plt.xlabel('Number of Simulations')
    plt.ylabel('KL Divergence')
    if save_figs:
        plt.savefig('figures/CMB_dkl.pdf')
    plt.show()

print(D_KL)

