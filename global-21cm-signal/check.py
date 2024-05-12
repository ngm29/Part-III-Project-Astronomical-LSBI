import numpy as np
import matplotlib.pyplot as plt
from globalemu.eval import evaluate
from scipy.stats import  gaussian_kde
from lsbi.model import LinearModel
from getdist import plots, MCSamples
from simulations import lsbi, lsbi_comp


# settings -------------------

d, n, N = 451, 7, 10000

#k_set = [1000,2000,5000,10000]
k_set = [10000]

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# compression
comp = 10

# number of relevant parameters
svd = False
c = 7

# seed
seed = 10000

# ----------------------------


# load data
data_dir = 'data/'
base_dir = 'T_release/'
params = np.loadtxt(data_dir + 'train_data.txt')
data = np.loadtxt(data_dir + 'train_labels.txt')

z = np.linspace(5, 50, 451)

# KL Divergence
D_KL = []

# posterior samples
samples_set = []

predictor = evaluate(base_dir=base_dir)

def twenty_one_cm_simulator(theta,comp):
    signal = predictor(theta)[0][::comp]
    noise = np.random.normal(0,25, size=len(signal))
    return signal + noise


np.random.seed(seed)
log_f_s = np.random.uniform(-0.32, -3.9)
v_c = np.random.uniform(4.3, 85)
log_f_x = np.random.uniform(-4, 1)
tau = np.random.uniform(0.05, 0.09)
alpha = np.random.uniform(1, 1.5)
log_nu = np.random.uniform(-1, 0.475)
R_mfp = np.random.uniform(10, 50)
theta_0 = np.transpose(np.array([10 ** log_f_s, v_c, 10 ** log_f_x, tau, alpha, 10 ** log_nu, R_mfp]))
D_0 = predictor(theta_0)[0]
D_0 = D_0[::comp]

# Discover significant parameters
k_0 = 1000
log_f_s = np.random.uniform(-0.32, -3.9, k_0)
v_c = np.random.uniform(4.3, 85, k_0)
log_f_x = np.random.uniform(-4, 1, k_0)
tau = np.random.uniform(0.05, 0.09, k_0)
alpha = np.random.uniform(1, 1.5, k_0)
log_nu = np.random.uniform(-1, 0.475, k_0)
R_mfp = np.random.uniform(10, 50, k_0)

theta = np.transpose(np.array([10 ** log_f_s, v_c, 10 ** log_f_x, tau, alpha, 10 ** log_nu, R_mfp]))

mu = np.mean(theta, axis=0)
Sigma = np.eye(n)*np.std(theta, axis=0) ** 2



for k in k_set:

    log_f_s = np.random.uniform(-0.32, -3.9, k)
    v_c = np.random.uniform(4.3, 85, k)
    log_f_x = np.random.uniform(-4, 1, k)
    tau = np.random.uniform(0.05, 0.09, k)
    alpha = np.random.uniform(1, 1.5, k)
    log_nu = np.random.uniform(-1, 0.475, k)
    R_mfp = np.random.uniform(10, 50, k)

    theta = np.transpose(np.array([10 ** log_f_s, v_c, 10 ** log_f_x, tau, alpha, 10 ** log_nu, R_mfp]))

    theta_0, D_0, theta, D, m, M, C, samps, samps2, d_kl = lsbi(d, n, k, N, comp=comp, c=n, theta=theta,
                                                                theta_0=theta_0,
                                                                mu=mu, Sigma=Sigma, simulator=twenty_one_cm_simulator,
                                                                seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))

    values = np.concatenate((theta, D), axis=1)

    joint = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma).joint()

    joint_samples = joint.rvs()

    pdfs1 = -np.log(joint.pdf(values, broadcast=True))

    pdfs2 = -np.log(joint.pdf(joint_samples, broadcast=True))

    n_bins = 100
    plt.hist(pdfs1, n_bins, (200,260 ), histtype='step', stacked=False, fill=True, label='Original simulations')
    plt.hist(pdfs2, n_bins, (200, 260), histtype='step', stacked=False, fill=False, label='Linear approximation')
    plt.legend(loc="upper right")
    plt.xlabel(r'$-\ln\mathcal{J}$')
    plt.ylabel('Frequency')
    plt.savefig('figures/21cm_goodness_of_fit.pdf')
    plt.show()










exit()


