import numpy as np
from scipy.special import polygamma, loggamma, digamma
from scipy import optimize

####################### functions to be optimized ###########################

def logdelta(x, epsilon, alpha, Delta_2sq, Delta_inf):
    g = (x-1)*Delta_inf
    vareps_lambda = 0.5*x*Delta_2sq*polygamma(1,alpha-g)
    return (x-1)*(vareps_lambda - epsilon)+(x-1)*np.log(x-1)-x*np.log(x)


#first derivative of f, for optimization
def Dlogdelta(x, epsilon, alpha, Delta_2sq, Delta_inf):
    g = (x-1)*Delta_inf
    return 0.5*x*Delta_2sq*polygamma(1,alpha-g) - epsilon+(x-1)*(0.5*Delta_2sq*polygamma(1,alpha-g)-0.5*x*Delta_2sq*Delta_inf*polygamma(2,alpha-g))+np.log(x-1)-np.log(x)


def epsilon_func(x, epsilon, lambda_ , Delta_2sq, Delta_inf):
    g = (lambda_-1)*Delta_inf
    return epsilon-0.5*lambda_*Delta_2sq*polygamma(1,x-g)


def Depsilon_func(x, epsilon, lambda_, Delta_2sq, Delta_inf):
    g = (lambda_-1)*Delta_inf
    return -0.5*lambda_*Delta_2sq*polygamma(2,x-g)


########################## privacy conversions ############################


#compute the privacy guarantee epsilon as in Theorem 1
def alpha2epsilon(min_alphas, lambda_ = 2, Delta_2sq = 1, Delta_inf = 1):
    g = (lambda_-1)*Delta_inf
    return 0.5*lambda_*Delta_2sq*polygamma(1,min_alphas-g)


#compute minimum alpha for a given epsilon
def epsilon2alpha(epsilon, lambda_ = 2, Delta_2sq = 1, Delta_inf = 1):
    alpha = optimize.fsolve(epsilon_func, \
                        x0 = 0.01+(lambda_-1)*Delta_inf, \
                        args = (epsilon, lambda_, Delta_2sq, Delta_inf), \
                        fprime=Depsilon_func)[0]
    return alpha


#translate from RDP to (epsilon-delta)-DP at a given epsilon
def alpha2adp(epsilon, alpha, Delta_2sq = 1, Delta_inf = 1):
    #minimize logdelta
    upper = alpha/Delta_inf+1
    soln = optimize.minimize(logdelta, x0 = upper-1e-5, \
                              jac=Dlogdelta, \
                              bounds = ((1+1e-5,upper),), \
                              args = (epsilon, alpha, Delta_2sq, Delta_inf))
    #returns both the optimal lambda and delta.
    return soln.x[0], np.exp(soln.fun[0]) 


########################### miscellaneous #############################


#Compute the exact value of epsilon using Dirichlet(x1+alpha) and Dirichlet(x2+alpha) according to equation (3)
def alpha2epsilon_evidence(x1, x2, alpha, lambda_):
    u1 = x1+alpha
    u2 = x2+alpha
    sumu1 = u1.sum()
    sumu2 = u2.sum()
    logE = (lambda_-1)*np.sum(loggamma(u2)-loggamma(u1)) \
            +np.sum(loggamma(u1+(lambda_-1)*(u1-u2))-loggamma(u1)) \
            -(lambda_-1)*(loggamma(sumu2)-loggamma(sumu1)) \
            -np.sum(loggamma(sumu1+(lambda_-1)*(sumu1-sumu2))-loggamma(sumu1))
    return logE/(lambda_-1) 


#Compute the KL-divergence between Dirichlet(a) and Dirichlet(b) 
def KLDir(a,b):
    #compute the KL-divergence between Dirichlet(a) and Dirichlet(b)
    a0 = np.sum(a, axis = 1)
    b0 = np.sum(b, axis = 1)
    KL = loggamma(a0)-np.sum(loggamma(a), axis = 1) \
                        -loggamma(b0)+np.sum(loggamma(b), axis = 1) \
                        +np.sum((a-b)*(digamma(a)-digamma(a0)[:, np.newaxis]), axis = 1)
    return KL
