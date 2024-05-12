import numpy as np
import matplotlib.pyplot as plt
from lsbi.model import LinearModel
from scipy.stats import chi2, gaussian_kde
from cmbemu.eval import evaluate
from getdist import plots, MCSamples
from simulations import lsbi, lsbi_comp


# settings -------------------

d, n, N = 300, 6, 10000

#k_set = [1000, 5000, 25000]
k_set = [10000]

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# compression
comp = 10

# seed
seed = 10000

# ----------------------------


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

# Compute the prior
mu = np.mean(params, axis=0)
Sigma = np.eye(n)*np.std(params, axis=0) ** 2


predictor = evaluate(base_dir=base_dir)


def CMB_simulator(theta,comp):
    signal = predictor(theta)[0][::comp] * chi2.rvs(2 * l[::comp] + 1) / (2 * l[::comp] + 1)
    return signal



for k in k_set:

    theta_0, D_0, theta, D, m, M, C, samps, samps2, d_kl = lsbi(d, n, k, N,  comp=comp, c=n,
                                                       mu=mu, Sigma=Sigma, simulator=CMB_simulator, seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


    values = np.concatenate((theta, D), axis=1)

    joint = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma).joint()

    joint_samples = joint.rvs()




    pdfs1 =  -np.log(joint.pdf(values, broadcast=True))

    pdfs2 = -np.log(joint.pdf(joint_samples, broadcast=True))

    n_bins = 100
    plt.hist(pdfs1, n_bins, (130,180), histtype='step', stacked=False, fill=True, label='Original simulations')
    plt.hist(pdfs2,n_bins,  (130,180), histtype='step', stacked=False, fill=False, label='Linear approximation')
    plt.legend(loc="upper right")
    plt.xlabel(r'$-\ln\mathcal{J}$')
    plt.ylabel('Frequency')
    plt.savefig('figures/CMB_goodness_of_fit.pdf')
    plt.show()





exit()


