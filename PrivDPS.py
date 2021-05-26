import numpy as np
from numpy.random import dirichlet
from scipy.special import polygamma, loggamma
from scipy import optimize

from utils import rho2alpha, alpha2rho


class DirichletPosteriorSampling():
    
    def __init__(self, rho, omega = 2, Delta_2sq = 1, Delta_inf = 1):
        self.rho = rho
        self.omega = omega
        self.Delta_2sq = Delta_2sq
        self.Delta_inf = Delta_inf
        self.gamma = self.omega*self.Delta_inf
        self.alpha = rho2alpha(self.rho, self.gamma, self.Delta_2sq)
        
    def sample(self, x, seed = None): #seed is only used to reproduce the results in the paper
        return np.random.default_rng(seed).dirichlet(x+self.alpha)
    
    def set_rho(self, new_rho):
        self.rho = new_rho
        self.alpha = rho2alpha(self.rho, self.gamma, self.Delta_2sq)
        
    def set_omega(self, new_omega):
        self.omega = new_omega
        self.gamma = self.omega*self.Delta_inf
        self.alpha = rho2alpha(self.rho, self.gamma, self.Delta_2sq)
    
    def get_alpha(self):
        return self.alpha
    


class GaussianMechanism():
    
    def __init__(self, rho, Delta_2sq = 1):
        self.rho = rho
        self.Delta_2sq = Delta_2sq
        self.sigma = np.sqrt(Delta_2sq/(2*self.rho))
        
    def add_noises(self, x, d, n_trials, seed = None): #x must be a (n_trials x d) array
        return x+np.random.default_rng(seed).normal(0, self.sigma, (n_trials,d))
    
    def set_rho(self, new_rho):
        self.rho = new_rho
        self.sigma = np.sqrt(Delta_2sq/(2*self.rho))
        
    def get_sigma(self):
        return self.sigma