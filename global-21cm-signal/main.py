import numpy as np
import matplotlib.pyplot as plt
from globalemu.eval import evaluate
from getdist import plots, MCSamples
from simulations import lsbi, lsbi_comp


# settings -------------------

d, n, N = 451, 7, 10000

k_set = [461,2000,10000,100000]

plot_prior = 1
plot_posterior = 1
plot_dkl = 0

# compression
comp = 2

# number of relevant parameters
svd = False
c = 7

# seed
seed = 1000

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


    theta_0, D_0, x_0,samps, samps2, d_kl = lsbi_comp(d, n, k, N, comp=comp, theta=theta, theta_0=theta_0,
                                                                          mu=mu, Sigma=Sigma, simulator=twenty_one_cm_simulator,
                                                                          seed=seed)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))


    """
    theta_0, D_0, theta, D, m, M, C, samps, samps2, d_kl = lsbi(d, n, k, N, c=c, comp=comp,  theta=theta, theta_0=theta_0,
                                                       mu=mu, Sigma=Sigma,
                                                       simulator=twenty_one_cm_simulator,
                                                       seed=seed, svd=svd)
    samples_set.append(samps2)
    D_KL.append(np.mean(d_kl))
    """


names = ["θ%s"%i for i in range(c)]
labels = [r'f_*',r'v_c',r' f_x',r'\tau',r'\alpha',r' \nu',r'R_{mfp}']
markers= {'θ0':theta_0[0],'θ1':theta_0[1],'θ2':theta_0[2],'θ3':theta_0[3],'θ4':theta_0[4], 'θ5':theta_0[5],'θ6':theta_0[6]}
samples = MCSamples(samples=samps,names = names, labels = labels, label='Prior')
samplesN = []
if plot_prior:
    samplesN.append(samples)
for i in range(len(k_set)):
    samplesN.append(MCSamples(samples=samples_set[i],names = names, labels = labels, label='%s Simulations'%k_set[i]))
#samplesN.append(MCSamples(samples=samples_set[1],names = names, labels = labels, label='Uncompressed'))
#samplesN.append(MCSamples(samples=samples_set[0],names = names, labels = labels, label='Compressed'))


if plot_posterior:
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 15
    g.settings.axes_labelsize = 15
    g.triangle_plot([samplesN[i] for i in range(len(samplesN))], markers=markers, filled=True, legend_loc='upper right',
                    contour_colors=['#006FED', 'gray', '#E03424', '#009966', '#000866', '#336600', 'm', 'r'])
    plt.savefig('figures/21cm_posterior.pdf')
    plt.show()

if plot_dkl:
    plt.semilogx(k_set, D_KL)
    plt.xlabel('Number of Simulations')
    plt.ylabel('KL Divergence')
    plt.savefig('figures/21cm_dkl.pdf')
    plt.show()

print(D_KL)