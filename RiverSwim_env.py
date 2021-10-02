import numpy as np

class RiverSwimEnv:
    LEFT_REWARD = 5.0 / 1000
    RIGHT_REWARD = 1.0

    def __init__(self, intermediate_states_count=4, max_steps=16):
        self._max_steps = max_steps
        self._current_state = None
        self._steps = None
        self._interm_states = intermediate_states_count
        self.reset()

    def reset(self):
        self._steps = 0
        self._current_state = 1
        return self._current_state, 0.0, False

    @property
    def n_actions(self):
        return 2

    @property
    def n_states(self):
        return 2 + self._interm_states

    def _get_transition_probs(self, action):
        if action == 0:
            if self._current_state == 0:
                return [0, 1.0, 0]
            else:
                return [1.0, 0, 0]

        elif action == 1:
            if self._current_state == 0:
                return [0, .4, .6]
            if self._current_state == self.n_states - 1:
                return [.4, .6, 0]
            else:
                return [.05, .6, .35]
        else:
            raise RuntimeError(
                "Unknown action {}. Max action is {}".format(action, self.n_actions))

    def step(self, action):
        """
        :param action:
        :type action: int
        :return: observation, reward, is_done
        :rtype: (int, float, bool)
        """
        reward = 0.0

        if self._steps >= self._max_steps:
            return self._current_state, reward, True

        transition = np.random.choice(
            range(3), p=self._get_transition_probs(action))
        if transition == 0:
            self._current_state -= 1
        elif transition == 1:
            pass
        else:
            self._current_state += 1

        if self._current_state == 0:
            reward = self.LEFT_REWARD
        elif self._current_state == self.n_states - 1:
            reward = self.RIGHT_REWARD

        self._steps += 1
        return self._current_state, reward, False

def one_episode(agent, env, episode_rewards, start_func):
    state, ep_reward, is_done = env.reset()
    start_func()
    t = 0
    while not is_done:
        action = agent.get_action(state)[t]
        
        next_state, reward, is_done = env.step(action)
        agent.update(state, action, reward, next_state)

        state = next_state
        ep_reward += reward
        t += 1

    episode_rewards.append(ep_reward)
    
def train_mdp_agent(agent, env, n_episodes):
    episode_rewards = []
    
    try:
        one_episode(agent, env, episode_rewards, agent.start_first_episode)
        episode_count = 1
    except AttributeError:
        episode_count = 0
        
    for ep in range(episode_count, n_episodes):
        one_episode(agent, env, episode_rewards, agent.start_episode)

    return episode_rewards
