import numpy as np
from scipy.special import polygamma, loggamma, digamma
from scipy import optimize

####################### functions to be optimized ###########################

def fhigh(x, delta, alpha, Delta_2sq, Delta_inf):
    return (x**2)*Delta_2sq*polygamma(1,alpha-x)/(Delta_inf**2)-np.log(1/delta)


# f in eq (7)
def flow(x, delta, alpha, Delta_2sq, Delta_inf):
    return Delta_2sq*polygamma(1,alpha-x)*(x/Delta_inf+1)+np.log(1/delta)*Delta_inf/x


#first derivative of f, for optimization
def Dflow(x, delta, alpha, Delta_2sq, Delta_inf):
    return -Delta_2sq*polygamma(2,alpha-x)*(x/Delta_inf+1) \
                +Delta_2sq*polygamma(1,alpha-x)/Delta_inf \
                -np.log(1/delta)*Delta_inf/(x**2)


def rho_func(x, rho, gamma, Delta_2sq):
    return rho-0.5*Delta_2sq*polygamma(1,x-gamma)


def Drho_func(x, rho, gamma, Delta_2sq):
    return -0.5*Delta_2sq*polygamma(2,x-gamma)


########################## privacy conversions ############################


#compute the privacy guarantee rho as in Theorem 1
def alpha2rho(min_alphas, gamma = 1, Delta_2sq = 1):
    return 0.5*Delta_2sq*polygamma(1,min_alphas-gamma)


#compute minimum alpha for a given rho
def rho2alpha(rho, gamma = 1, Delta_2sq = 1):
    alpha = optimize.fsolve(rho_func, \
                        x0 = 0.01+gamma, \
                        args = (rho, gamma, Delta_2sq), \
                        fprime=Drho_func)[0]
    return alpha


#translate from tCDP to (epsilon-delta)-DP at a given delta
def tcdp2adp(delta, alpha, Delta_2sq = 1, Delta_inf = 1):
    #solve for gamma_M
    gamma_M = optimize.brentq(fhigh, \
                               a = 0, \
                               b = alpha, \
                               args = (delta, alpha, Delta_2sq, Delta_inf))
    
    #minimize f
    soln = optimize.minimize(flow, x0 = gamma_M, \
                              jac=Dflow, \
                              bounds = ((10e-5,gamma_M),), \
                              args = (delta, alpha, Delta_2sq, Delta_inf))
    #returns both the optimal gamma and epsilon.
    return soln.x[0], soln.fun[0] 


########################### miscellaneous #############################


#Compute the exact value of rho using Dirichlet(x1+alpha) and Dirichlet(x2+alpha) according to equation (3)
def alpha2rho_evidence(x1, x2, alpha, omega):
    u1 = x1+alpha
    u2 = x2+alpha
    sumu1 = u1.sum()
    sumu2 = u2.sum()
    logE = (omega-1)*np.sum(loggamma(u2)-loggamma(u1)) \
            +np.sum(loggamma(u1+(omega-1)*(u1-u2))-loggamma(u1)) \
            -(omega-1)*(loggamma(sumu2)-loggamma(sumu1)) \
            -np.sum(loggamma(sumu1+(omega-1)*(sumu1-sumu2))-loggamma(sumu1))
    return logE/(omega*(omega-1)) 


#Compute the KL-divergence between Dirichlet(a) and Dirichlet(b) 
def KLDir(a,b):
    #compute the KL-divergence between Dirichlet(a) and Dirichlet(b)
    a0 = np.sum(a, axis = 1)
    b0 = np.sum(b, axis = 1)
    KL = loggamma(a0)-np.sum(loggamma(a), axis = 1) \
                        -loggamma(b0)+np.sum(loggamma(b), axis = 1) \
                        +np.sum((a-b)*(digamma(a)-digamma(a0)[:, np.newaxis]), axis = 1)
    return KL
