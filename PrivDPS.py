import numpy as np
from numpy.random import dirichlet
from scipy.special import polygamma, loggamma
from scipy import optimize

from utils import epsilon2alpha, alpha2epsilon


class DirichletPosteriorSampling():
    
    def __init__(self, epsilon, lambda_ = 2, Delta_2sq = 1, Delta_inf = 1):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_2sq = Delta_2sq
        self.Delta_inf = Delta_inf
        self.alpha = epsilon2alpha(self.epsilon, self.lambda_, self.Delta_2sq, self.Delta_inf)
        
    def sample(self, x, seed = None): #seed is only used to reproduce the results in the paper
        return np.random.default_rng(seed).dirichlet(x+self.alpha)
    
    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.alpha = epsilon2alpha(self.epsilon, self.lambda_, self.Delta_2sq, self.Delta_inf)
        
    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda
        self.alpha = epsilon2alpha(self.epsilon,  self.lambda_, self.Delta_2sq, self.Delta_inf)
    
    def get_alpha(self):
        return self.alpha
    


class GaussianMechanism():
    
    def __init__(self, epsilon, lambda_, Delta_2sq = 1):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_2sq = Delta_2sq
        self.sigma = np.sqrt(self.lambda_*self.Delta_2sq/(2*self.epsilon))
        
    def add_noises(self, x, d, n_trials, seed = None): #x must be a (n_trials x d) array
        return x+np.random.default_rng(seed).normal(0, self.sigma, (n_trials,d))
    
    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
        self.sigma = np.sqrt(self.lambda_*Delta_2sq/(2*self.epsilon))
        
    def get_sigma(self):
        return self.sigma


class LaplaceMechanism():
    
    def __init__(self, epsilon, lambda_, Delta_1 = 1):
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.Delta_1 = Delta_1
        self.scale = Delta_1*np.sqrt(self.lambda_/(2*self.epsilon))
        
    def add_noises(self, x, d, n_trials, seed = None): #x must be a (n_trials x d) array
        return x+np.random.default_rng(seed).laplace(0, self.scale, (n_trials,d))
    
    def set_rho(self, new_epsilon):
        self.epsilon = new_epsilon
        self.scale = Delta_1*np.sqrt(self.lambda_/(2*self.epsilon))
        
    def get_scale(self):
        return self.scale
