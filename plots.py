import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import pyplot

from PrivDPS import DirichletPosteriorSampling, GaussianMechanism, LaplaceMechanism
from utils import alpha2epsilon, epsilon2alpha, alpha2adp, alpha2epsilon_evidence, KLDir
from RiverSwim_env import *
from PrivPSRL import *

#matplotlib parameters
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('plot_style.txt')
alphaVal = 0.8
linethick = 0.7
color1 = (228/255, 26/255, 28/255)
color2 = (55/255, 126/255, 184/255)
color3 = (77/255, 175/255, 74/255)
colors = [color1,color2,color3]



fig, axes = plt.subplots(nrows = 1, ncols = 2)
ax0 = axes[0]
ax1 = axes[1]

###################### Plot of RDP guarantees ##########################

d = 10
n_points = 80 #number of points in the plots

# uniform prior
alphas = np.linspace(2,20,n_points)
x1 = np.array([ 11 , 8 , 65 , 25 , 38 , 0])
x2 = np.array([ 11 , 8 , 65 , 25 , 38 , 1])

Delta_2sq = 1
Delta_inf = 1
lambda_ = 2
gamma = (lambda_-1)*Delta_inf

true_epsilon = np.zeros(n_points)

for i,alpha in enumerate(alphas):
    true_epsilon[i] = alpha2epsilon_evidence(x1, x2, alpha, lambda_)

upper_epsilon = alpha2epsilon(alphas, lambda_, Delta_2sq, Delta_inf)

ax0.plot(alphas,true_epsilon,
         lw=linethick,
         label='True $\\varepsilon$',
         alpha=alphaVal)
ax0.plot(alphas,upper_epsilon,'--',
         linewidth=linethick,
         label='$(2,\\varepsilon)$-RDP upper bound',
         alpha=alphaVal)

legend = ax0.legend(loc='upper right', prop={'size': 5.5}, framealpha=0.7)
frame = legend.get_frame().set_linewidth(0.0)
ax0.set_yscale('log') 
ax0.set_xlim(1.5, 20)
ax0.set_ylim(0.03, 1)
ax0.set_xticks(np.arange(2, 20, 4))
ax0.tick_params(axis='both', which='major', labelsize=5)
ax0.set_xlabel('$\\alpha$',fontsize=6, labelpad=0)
ax0.set_ylabel('$\\epsilon$',fontsize=6, labelpad=0)
    


################## Plot of Approximate DP guarantees ##################


epsilons = np.linspace(0,25,n_points)
N=100
Delta_inf = 1
Delta_2sq = 1

def plot_epsilon_delta(alpha, label):
    deltas = np.zeros(n_points)

    for i,epsilon in enumerate(epsilons):
        deltas[i] = alpha2adp(epsilon, alpha, Delta_2sq, Delta_inf)[1]


    ax1.plot(epsilons,deltas, 
            linestyle = '-',
            lw=linethick,
            label=label,
            alpha=alphaVal)

plot_epsilon_delta(2, '$\\alpha = 2$')
plot_epsilon_delta(5, '$\\alpha = 5$')
plot_epsilon_delta(10, '$\\alpha = 10$')

legend = ax1.legend(loc='upper right', prop={'size': 5.5}, framealpha=0.7)
frame = legend.get_frame().set_linewidth(0.0)
ax1.set_yscale('log')
ax1.set_xlim(0, 20)
ax1.set_ylim(10e-11, 1)
ax1.set_xticks(np.arange(0, 24, 4))
ax1.tick_params(axis='both', which='major', labelsize=5)
ax1.set_xlabel("$\\varepsilon$", fontsize = 6, labelpad=0)
ax1.set_ylabel('$\\delta$', fontsize = 6, labelpad=0)

fig.set_size_inches(3.5, 1.6)
fig.tight_layout()
plt.savefig('Dirichlet_guarantees.pdf', format='pdf', dpi=600, bbox_inches='tight', transparent=True)



########## Plots of mechanisms for private normalized histograms ##########

seed = 1122
n_trials = 200 #number of trials at each N, epsilon and d
ds = [10,1000]
eta = 5
Ns = np.logspace(1,5,10).astype(np.uint32)
epsilons = [0.01, 0.1, 1]
Delta_inf = 1
Delta_2sq = 2
Delta_1 = 2
lambda_ = 2
#g = (lambda_-1)*Delta_inf



eps_dir = np.zeros(Ns.shape[0]) 
eps_gauss = np.zeros(Ns.shape[0])
eps_laplace = np.zeros(Ns.shape[0]) 

eps_dir_err = np.zeros(Ns.shape[0]) 
eps_gauss_err = np.zeros(Ns.shape[0]) 
eps_laplace_err = np.zeros(Ns.shape[0]) 

fig, axes = plt.subplots(nrows = 2 , ncols = 3)

for k,d in enumerate(ds):                #for each d
    for l,epsilon in enumerate(epsilons):        #for each epsilon 
        for i,N in enumerate(Ns):        #for each N
            
            #generate p for n_trials times
            prob = np.random.default_rng(seed).dirichlet([eta]*d)
            x = np.random.default_rng(seed).multinomial(N,prob,n_trials) 
            p = x/N

            ################################################

            DPS = DirichletPosteriorSampling(epsilon, lambda_, Delta_2sq, Delta_inf)
            q_dir = np.array([DPS.sample(x[row]) for row in range(n_trials)])
            dirVec = np.sqrt(np.sum((p-q_dir)**2,axis=1))
            eps_dir[i] = dirVec.mean()             #point estimate
            eps_dir_err[i] = 2*dirVec.std()        #error bar

            ###############################################
            
            GM = GaussianMechanism(epsilon, lambda_, Delta_2sq/(N**2)) #note the l2-sensitivity is Delta_2/N
            q_gauss = GM.add_noises(p, d, n_trials)
            gaussVec = np.sqrt(np.sum((p-q_gauss)**2,axis=1))
            eps_gauss[i] = gaussVec.mean()         #point estimate
            eps_gauss_err[i] = 2*gaussVec.std()    #error bar

            ################################################

            LM = LaplaceMechanism(epsilon, lambda_, Delta_1/N) #note the l2-sensitivity is Delta_1/N
            q_laplace = LM.add_noises(p, d, n_trials)
            laplaceVec = np.sqrt(np.sum((p-q_laplace)**2,axis=1))
            eps_laplace[i] = laplaceVec.mean()         #point estimate
            eps_laplace_err[i] = 2*laplaceVec.std()    #error bar

            ################################################
            
        ax = axes[k,l]

        ax.errorbar(Ns,eps_dir,yerr = eps_dir_err, 
                    color=color1,
                    linestyle = '-',
                    lw=linethick,
                    label='Dirichlet',
                    alpha=alphaVal)
        ax.errorbar(Ns,eps_gauss, yerr = eps_gauss_err,
                    color=color2,
                    linestyle = '-',
                    lw=linethick,
                    label='Gaussian',
                    alpha=alphaVal)
        ax.errorbar(Ns,eps_laplace, yerr = eps_laplace_err,
                    color=color3,
                    linestyle = '-',
                    lw=linethick,
                    label='Laplace',
                    alpha=alphaVal)

        legend = ax.legend(loc='upper right', prop={'size': 4.5}, framealpha=0.7)
        frame = legend.get_frame().set_linewidth(0.0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 10**5)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_title(r'$d = '+str(d)+r', \varepsilon = '+ str(epsilon)+ '$', fontsize = 6)
        ax.set_xlabel("Sample size ("+r'$N$'+")", fontsize = 5, labelpad=0)
        ax.set_ylabel('$\\ell^{2}$', fontsize = 5, labelpad=0)

fig.set_size_inches(4.5, 2.25)
fig.tight_layout()
plt.savefig('private_normalized_histograms.pdf', format='pdf', dpi=600, bbox_inches='tight', transparent=True)



########## Plots of mechanisms for private Multinomial-Dirichlet sampling ##########

seed = 2233
n_trials = 200 #number of trials at each N, epsilon and eta
d = 20
etas = [0.1,1,10]
Ns = np.logspace(1,5,10).astype(np.uint32)
epsilons = [0.01, 0.1, 1]
Delta_inf = 1
Delta_2sq = 2
lambda_ = 2

kl_dir = np.zeros(Ns.shape[0]) 
kl_gauss = np.zeros(Ns.shape[0]) 

kl_dir_err = np.zeros(Ns.shape[0]) 
kl_gauss_err = np.zeros(Ns.shape[0]) 

fig, axes = plt.subplots(nrows = 1 , ncols = 3)

for l,epsilon in enumerate(epsilons):           #for each epsilon 
    for k,eta in enumerate(etas):       #for each eta
        for i,N in enumerate(Ns):       #for each N
            
            #generate x for n_trials times
            prob = np.random.default_rng(seed).dirichlet([eta]*d, size = n_trials)
            x = np.array([np.random.default_rng(seed).multinomial(N,prob[i]) for i in range(n_trials)])

            ################################################

            alpha = epsilon2alpha(epsilon, gamma, Delta_2sq)
            ai = x+1
            bi = x+alpha
            DirKLVec = KLDir(ai,bi)
            kl_dir[i] = DirKLVec.mean()
            kl_dir_err[i] = 2*DirKLVec.std()
            
            ################################################
        
        ax = axes[l]

        ax.errorbar(Ns,kl_dir, yerr= kl_dir_err,
                    color=colors[k],
                    linestyle = '-',
                    lw=linethick,
                    label=r'$\eta = '+str(eta)+ '$',
                    alpha=alphaVal)

        legend = ax.legend(loc='lower left', prop={'size': 4.5}, framealpha=0.7)
        frame = legend.get_frame().set_linewidth(0.0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10, 10**5)
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_title(r'$\varepsilon = '+ str(epsilon)+ '$', fontsize = 6)
        ax.set_xlabel("Sample size ("+r'$N$'+")", fontsize = 5, labelpad=0)
        ax.set_ylabel('KL-divergence', fontsize = 5, labelpad=0)

fig.set_size_inches(4.5, 1.16)
fig.tight_layout()
plt.savefig('Multinomial-Dirichlet-sampling.pdf', format='pdf', dpi=600, bbox_inches='tight', transparent=True)

###################### Plot of private PSRL with diffuse and concentrated sampling #######################

N=3000
horizon = 30
ALPHA = 10
lambda_ = 2
PLOT_ALPHA = 0.02
alphaVal = 0.8
linethick = 0.5

moving_average = lambda x, **kw: DataFrame(
    {'x': np.asarray(x)}).x.ewm(**kw).mean().values

fig, axes = plt.subplots(nrows = 2 , ncols = 2, figsize = (4.5,2.6) )

for epsilon_plot in zip([0.01,0.1,1,10],axes.flat):

    epsilonVal = epsilon_plot[0]
    ax = epsilon_plot[1]

    for j,agent_name in enumerate([(PsrlAgent,'Non-private'), 
                                   (DiffusePsrlAgent, 'Diffuse'),  
                                   (ConcentratedPsrlAgent, 'Concentrated')]):

        env = RiverSwimEnv(max_steps=horizon)
        agent = agent_name[0](env.n_states, env.n_actions, horizon=horizon, \
                          alpha=ALPHA, epsilon=epsilonVal, lambda_=lambda_, episodes=N)
        rews = train_mdp_agent(agent, env, N)
        if j==1 or j==2:
            print("cumulative epsilon =", agent._sum_epsilon)


        ax.plot(moving_average(np.array(rews), alpha=PLOT_ALPHA), color=colors[j],
                        linestyle = '-',
                        lw=linethick,
                        label=agent_name[1],
                        alpha=alphaVal)
    
    legend = ax.legend(loc='upper left', prop={'size': 4.5}, framealpha=0.7)
    frame = legend.get_frame().set_linewidth(0.0)
    ax.set_xlim(0, N)
    ax.set_ylim(0, 10)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.set_title(r'$\varepsilon = '+str(epsilonVal)+'$', fontsize=6)
    ax.set_xlabel("Episode count", fontsize = 5, labelpad=0)
    ax.set_ylabel("Reward", fontsize = 5, labelpad=0)
fig.tight_layout()
plt.savefig('PrivPSRL.pdf', format='pdf', dpi=600, bbox_inches='tight', transparent=True)
