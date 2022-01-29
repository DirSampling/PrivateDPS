import numpy as np
from scipy.special import polygamma
from scipy import optimize


def concentrated_objective(x,epsilon,lambda_,n):
    return epsilon-2*n*lambda_*polygamma(1,x-lambda_+1)

def concentrated_objective_derivative(x,epsilon,lambda_,n):
    return [-2*n*lambda_*polygamma(2,x-lambda_+1)]

def sample_normal_gamma(mu, lmbd, alpha, beta):
    """ https://en.wikipedia.org/wiki/Normal-gamma_distribution
    """
    tau = np.random.gamma(alpha, beta)
    mu = np.random.normal(mu, 1.0 / np.sqrt(lmbd * tau))
    return mu, tau


class PsrlAgent:
    def __init__(self, n_states, n_actions, alpha=10.0, 
                 horizon=10, epsilon = None, lambda_=None, episodes=100):
        self._n_states = n_states
        self._n_actions = n_actions
        self._horizon = horizon
        self._episodes = episodes
        self._episode_count = 0
        self._alpha = alpha #prior parameter
        self._epsilon = None #privacy parameter (epsilon) for posterior Dirichlet sampling
        self._lambda_ = None #privacy parameter (lambda_) for posterior Dirichlet sampling
        self._r = 1 #diffuse parameter, 1 in non-private case

        # params for transition sampling - Dirichlet distribution
        # Each row sums to one.
        self._dirichlet_params = np.zeros(
            (n_states, n_states, n_actions)) + self._alpha

        # params for reward sampling - Normal-gamma distribution
        self._mu_matrix = np.zeros((n_states, n_actions)) + 1.0
        self._state_action_counts = np.zeros(
            (n_states, n_actions)) + 1.0  # lambda

        self._alpha_matrix = np.zeros((n_states, n_actions)) + 1.0
        self._beta_matrix = np.zeros((n_states, n_actions)) + 1.0

    def value_iteration(self, P, R):
        V = np.zeros((self._horizon + 1, self._n_states))
        Q = np.zeros((self._horizon + 1, self._n_states * self._n_actions))        

        for t in np.arange(self._horizon - 1, -1, -1):
            PV = np.zeros((self._n_states,self._n_actions))
            for i in range(self._n_actions):
                PV[:,i] = P[:,:,i]@V[t+1,:]
            Q_matrix = R + PV
            V[t, :] = np.max(Q_matrix, axis = 1)

        return V

    def start_episode(self):
        self._episode_count += 1
        
        # sample new  mdp
        self._transition_matrix = np.apply_along_axis(
            np.random.dirichlet, 1, self._dirichlet_params)
        
        self._alpha_matrix = 1.0 + 0.5*(self._state_action_counts - 1.0)
        
        
        R_mus, R_stds = sample_normal_gamma(
            self._mu_matrix,
            self._state_action_counts,
            self._alpha_matrix,
            self._beta_matrix
        )

        self._rewards = R_mus
        self._current_value_function = self.value_iteration(
            self._transition_matrix, self._rewards)
        
    def get_action(self, state):
        return np.argmax(self._rewards[state] +
                         self._current_value_function @ self._transition_matrix[state], axis = 1)

    def update(self, state, action, reward, next_state):
        self._dirichlet_params[state][next_state][action] += self._r
        
        count = self._state_action_counts[state][action]-1
        mu = self._mu_matrix[state][action]

        if count==0:
            old_avg = 0
        else:
            old_avg = (mu*(1+count)-1)/count
        
        new_avg = (mu*(1+count)-1+reward)/(count+1)
        self._mu_matrix[state][action] = (1+(count+1)*new_avg)/(2+count)
        beta = self._beta_matrix[state][action] 
        self._beta_matrix[state][action] = beta \
                                + 0.5*(reward**2+count*(old_avg**2-new_avg**2)) 

        self._state_action_counts[state][action] += 1
        

    def get_q_matrix(self):
        return self._rewards + self._current_value_function @ self._transition_matrix
    
class PsrlAgentWithNoisyRewards(PsrlAgent):
    def __init__(self, n_states, n_actions, alpha=10.0, 
                 horizon=10, epsilon = None, lambda_=None, ng_epsilon = 0.5, episodes=100):
        super().__init__(n_states, n_actions, alpha, 
                 horizon, epsilon, lambda_, episodes)
        
        self._ng_epsilon = ng_epsilon #privacy parameter (epsilon) of Gaussian mechanism
        
        #Find sigma for the Gaussian mechanism
        self._scale = np.sqrt(3*self._episodes)/np.sqrt(2*self._ng_epsilon)
        
        
    def start_first_episode(self):
        super().start_episode()

        
    def start_episode(self):

        # sample new  mdp
        self._transition_matrix = np.apply_along_axis(
            np.random.dirichlet, 1, self._dirichlet_params)
        
        self._alpha_matrix = 1.0 + 0.5*(self._state_action_counts - 1.0)
        
        
        #Noises for the Normal-Gamma parameters
        n_eps = np.random.normal(loc=0, 
                                scale=self._scale,
                                size = (self._n_states, self._n_actions))
        sum_eps = np.random.normal(loc=0, 
                                 scale=self._scale,
                                  size = (self._n_states, self._n_actions))

        beta_eps = np.random.normal(loc=0, 
                                 scale=self._scale,
                                   size = (self._n_states, self._n_actions))




        sample_sum = self._mu_matrix*self._state_action_counts-1

        noisy_sum = sample_sum + sum_eps

        #Apply Gaussian mechanism to all Normal-Gamma parameters
        noisy_count = np.floor(np.clip(self._state_action_counts-1 + n_eps,0,None))+1
        noisy_mus = np.clip((1+noisy_sum)/noisy_count,0,None)
        noisy_alphas = 1.0 + 0.5*(noisy_count - 1.0)
        noisy_betas = np.clip(self._beta_matrix-1 + beta_eps,0,None)+1 \
                         + 0.5*((noisy_count-1) * (noisy_sum/noisy_count - 1)**2)/noisy_count 
        R_mus, R_stds = sample_normal_gamma(
            noisy_mus,
            noisy_count,
            noisy_alphas,
            noisy_betas
        )

        self._rewards = R_mus
        self._current_value_function = self.value_iteration(
            self._transition_matrix, self._rewards)
    
class DiffusePsrlAgent(PsrlAgentWithNoisyRewards):
    def __init__(self, n_states, n_actions, alpha=10.0, 
                 horizon=10, epsilon=0.1, lambda_=6.0, ng_epsilon=0.5, episodes=100):
        super().__init__(n_states, n_actions, alpha, 
                 horizon, epsilon, lambda_, ng_epsilon, episodes)
        
        self._epsilon = epsilon
        self._lambda_ = lambda_    
        self._gamma = self._lambda_-1
        
        if self._gamma >= self._alpha:
            raise ValueError(f"gamma = {self._gamma} is greater or equal to alpha = {self._alpha}")
            
            
        #dont forget to divide by self._episodes 
        self._r = np.sqrt(2*self._epsilon/(4*self._episodes*self._lambda_*polygamma(1,self._alpha-self._lambda_+1)))
        self._sum_epsilon = 2*(self._r**2)*(self._lambda_*polygamma(1,self._alpha-self._lambda_+1)) #tracks the privacy budget
        print("r =", self._r)
        
        
    def start_first_episode(self):
        self._sum_epsilon += 2*(self._r**2)*(self._lambda_*polygamma(1,self._alpha-self._lambda_+1))
        super().start_first_episode()
        
        
    def start_episode(self):
        self._sum_epsilon += 2*(self._r**2)*(self._lambda_*polygamma(1,self._alpha-self._lambda_+1))
        super().start_episode()
        
        
class ConcentratedPsrlAgent(PsrlAgentWithNoisyRewards):
    def __init__(self, n_states, n_actions, alpha=1.0, 
                 horizon=10, epsilon=0.1, lambda_=6.0, ng_epsilon=0.5, episodes=100):
        super().__init__(n_states, n_actions, alpha, 
                 horizon, epsilon, lambda_, ng_epsilon, episodes)
        
        self._epsilon = epsilon
        self._lambda_ = lambda_
        
        if self._lambda_ >= self._alpha+1:
            raise ValueError(f"lambda_ = {self._lambda_} is greater or equal to alpha + 1 = {self._alpha+1}")        
        
        
        sol = optimize.fsolve(concentrated_objective, \
                            x0 = self._lambda_, \
                            args = (self._epsilon,self._lambda_,self._episodes), \
                            fprime=concentrated_objective_derivative)[0]
        self._alpha = sol
        print('alpha =',self._alpha)
            
        self._dirichlet_params = np.zeros(
            (n_states, n_states, n_actions)) + self._alpha #use private prior instead
        self._sum_epsilon = 2*self._lambda_*polygamma(1,self._alpha-self._lambda_+1) #tracks the privacy budget
        
        
    def start_first_episode(self):
        self._sum_epsilon += 2*self._lambda_*polygamma(1,self._alpha-self._lambda_+1)
        super().start_first_episode()
        
        
    def start_episode(self):
        self._sum_epsilon += 2*self._lambda_*polygamma(1,self._alpha-self._lambda_+1)
        super().start_episode()
