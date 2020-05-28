import numpy as np
from collections import deque
import tiles3


class TileEncoder():
    def __init__(self, var_ranges, num_tiles, num_tilings):
        assert len(var_ranges) == len(num_tiles), \
            "Input variables length do not match"
        assert all(isinstance(val, int) and val > 0 for val in num_tiles), \
            "number of tiles should be an array of integers > 0"
        assert all(len(var_range) == 2 for var_range in var_ranges), \
            "variable range should be a finite numeric interval"
        assert isinstance(num_tilings, int), \
            "number of tilings should be an integer" 
        self.var_ranges = var_ranges
        self.num_tiles = num_tiles
        self.num_var = len(var_ranges)
        self.num_tilings = num_tilings
        self.var_coeff = np.zeros(self.num_var, dtype=np.float32)
        self.get_coeffs()
        self.iht_size = self.calc_iht_size()
        self.iht = tiles3.IHT(self.iht_size)

    def get_coeffs(self):
        """Calculate the coefficients for scaling the inputs"""
        for i, var_range in enumerate(self.var_ranges):
            self.var_coeff[i] = np.floor(self.num_tiles[i] / np.abs(var_range[1] - var_range[0]))

    def calc_iht_size(self):
        iht_size = 1
        for num_tiles in self.num_tiles:
            iht_size *= (5*num_tiles)
        # print("Hash table index range is {}".format(iht_size*3))
        return iht_size

    def get_feature(self, values):
        assert len(values) == self.num_var, "Incorrect input length"
        new_values = np.array(values, dtype=np.float32) * self.var_coeff
        return tiles3.tiles(self.iht, self.num_tilings, new_values)


class BaseAgent():
    """
    Base class for implementing classic RL agents
    """
    def __init__(self, obs_limits, num_actions, n_step=1, epsilon=0.1, step_size=0.5,
                 discount=1, seed=None, decay_factor=None, num_tilings=8, num_tiles=None):
        # set seed if provided
        self.seed = seed
        self.policy_rand_generator = np.random.default_rng(self.seed)
        # setup tile encoding
        if num_tiles == None:
            num_tiles = [8] * len(obs_limits)
        self.tc = TileEncoder(obs_limits, num_tiles, num_tilings)
        # set agent parameters
        self.n_step = n_step
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.discount = discount
        self.step_size = step_size/self.tc.num_tilings
        self.d_step_size = self.step_size
        self.decay_factor = decay_factor
        self.value_function = np.zeros((self.num_actions, self.tc.iht_size, self.tc.num_tilings),
                                       dtype=np.float32)
        self.indices = [i for i in range(self.tc.num_tilings)]
        self.episode = 0
        # create experience buffers
        self.state_buffer = deque()
        self.action_buffer = deque()
        self.reward_buffer = deque()
        self.time_step = 0
        self.disc_powers = [np.power(self.discount, i) for i in range(self.n_step)]
        self.disc_n_power = np.power(self.discount, self.n_step)

    def agent_policy(self, state):
        """
        Return an action according to the policy given a state
        """
        if self.policy_rand_generator.random() < self.epsilon:
            # Take random exploratory action
            return self.policy_rand_generator.integers(0, self.num_actions)
        else:
            # Take greedy action w.r.to current value function
            action_vals= np.sum(self.value_function[:, self.tc.get_feature(state), self.indices], axis=1)
            max_val = np.amax(action_vals)
            max_actions = np.where(action_vals == max_val)[0]
            return self.policy_rand_generator.choice(max_actions)

    def start(self, state):
        """ Start the agent for the episode """
        self.episode += 1
        self.state_buffer.append(state)
        self.action_buffer.append(self.agent_policy(state))
        return self.action_buffer[0]
    
    def update_value(self, terminal=False):
        raise NotImplementedError

    def take_step(self, reward, state):
        """
        Agent updates the state values and returns the next action
        """
        self.time_step += 1
        action = self.agent_policy(state)
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        # Perform value function update if enough experience is available
        self.update_value(terminal=False)
        return action

    def end(self, reward):
        """
        Terminate the episode and update the values
        """
        self.time_step += 1
        self.reward_buffer.append(reward)
        self.update_value(terminal=True)
        self.reset_episode()

    def reset_episode(self):
        """ Resets episode related parameters of the agent """
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.time_step = 0
        if self.decay_factor:
            self.d_step_size = self.d_step_size * self.decay_factor


class SARSA_agent(BaseAgent):
    """
    n-step semi gradient SARSA agent for episodic tasks with discrete action 
    space and continuous state space
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_value(self, terminal):
        """Performs an update of the value funciton"""
        update_done = False
        update_timestep = self.time_step - self.n_step
        # Check if experience is available for update
        while ((update_timestep >= 0 and not update_done) or
                (terminal and len(self.reward_buffer) > 0)):
            update_done = True
            # Find the sampled return
            sampled_returns = 0
            for i in range(len(self.reward_buffer)):
                sampled_returns += self.disc_powers[i] * self.reward_buffer[i]
            # Find TD target
            update_state_fv = self.tc.get_feature(self.state_buffer[0])
            if terminal:
                target = sampled_returns
            else:
                state = self.state_buffer[-1]
                action = self.action_buffer[-1]
                fv = self.tc.get_feature(state)
                target = sampled_returns + self.disc_n_power * np.sum(self.value_function[action, fv, self.indices])
            # Perform TD update
            td_error = target - np.sum(self.value_function[self.action_buffer[0], update_state_fv, self.indices])
            self.value_function[self.action_buffer[0], update_state_fv, self.indices] += self.d_step_size * td_error
            self.state_buffer.popleft()
            self.action_buffer.popleft()
            self.reward_buffer.popleft()


class Expected_SARSA_agent(BaseAgent):
    """
    n-step expected semi gradient SARSA agent for episodic tasks with discrete
    action space and continuous state space
    """ 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_value(self, terminal):
        """Performs an update of the value funciton"""
        update_done = False
        update_timestep = self.time_step - self.n_step
        # Check if experience is available for update
        while ((update_timestep >= 0 and not update_done) or
                (terminal and len(self.reward_buffer) > 0)):
            update_done = True
            # Find the sampled return
            sampled_returns = 0
            for i in range(len(self.reward_buffer)):
                sampled_returns += self.disc_powers[i] * self.reward_buffer[i]
            # Find TD target
            update_state_fv = self.tc.get_feature(self.state_buffer[0])
            if terminal:
                target = sampled_returns
            else:
                state = self.state_buffer[-1]
                fv = self.tc.get_feature(state)
                weights = np.zeros(self.num_actions, dtype=np.float32)
                action_values = np.sum(self.value_function[:, fv, self.indices], axis=1)
                max_val = np.amax(action_values)
                max_actions = np.where(action_values == max_val)[0]
                weights[max_actions] = (1-self.epsilon)/max_actions.size
                weights += self.epsilon/self.num_actions
                target = sampled_returns + self.disc_n_power * np.sum(weights * action_values)
            # Perform TD update
            td_error = target - np.sum(self.value_function[self.action_buffer[0], update_state_fv, self.indices])
            self.value_function[self.action_buffer[0], update_state_fv, self.indices] += self.d_step_size * td_error
            self.state_buffer.popleft()
            self.action_buffer.popleft()
            self.reward_buffer.popleft()


class Q_agent(BaseAgent):
    """
    Q-Learning agent for episodic tasks with discrete
    action space and continuous state space
    
    Note: Agent can perform only one step updates 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_value(self, terminal):
        """Performs an update of the value funciton"""
        update_state_fv = self.tc.get_feature(self.state_buffer[0])
        if terminal:
            target = self.reward_buffer[-1]
        else:
            state = self.state_buffer[-1]
            fv = self.tc.get_feature(state)
            max_action_value = np.amax(np.sum(self.value_function[:, fv, self.indices], axis=1))
            target = self.reward_buffer[-1] + self.discount * max_action_value
            
        td_error = target - np.sum(self.value_function[self.action_buffer[0], update_state_fv, self.indices])
        self.value_function[self.action_buffer[0], update_state_fv, self.indices] += self.d_step_size * td_error
        self.state_buffer.popleft()
        self.action_buffer.popleft()
        self.reward_buffer.popleft()

class TO_SARSA_Lambda_agent(BaseAgent):
    """
    Agent that learns using True Online SARSA (Lambda) Agent
    """
    def __init__(self, *args, lambda_val=0.98, **kw_args):
        super().__init__(*args, **kw_args)
        self.lambda_val = lambda_val

    def start(self, state):
        """ Start the agent for the episode """
        action = super().start(state)
        self.trace = np.zeros_like(self.value_function, dtype=np.float32)
        self.Q_Old = 0
        return action

    def update_value(self, terminal):
        """Performs an update of the value funciton"""
        # Calculate TD error
        fv = self.tc.get_feature(self.state_buffer[0])
        Q = np.sum(self.value_function[self.action_buffer[0], fv, self.indices])
        if terminal:
            Q_dash = 0
        else:
            fv_dash = self.tc.get_feature(self.state_buffer[1])
            Q_dash = np.sum(self.value_function[self.action_buffer[1], fv_dash, self.indices])
        td_error = (self.reward_buffer[0] + self.discount * Q_dash) - Q
        # Estimate Dutch's trace
        multiplier = self.step_size * self.discount * self.lambda_val
        multiplier = multiplier * np.sum(self.trace[self.action_buffer[0], fv, self.indices])
        self.trace *= self.lambda_val * self.discount
        self.trace[self.action_buffer[0], fv, self.indices] += (1 - multiplier)
        # Update value function
        self.value_function += self.step_size * (td_error + Q - self.Q_Old) * self.trace
        self.value_function[self.action_buffer[0], fv, self.indices] -= self.step_size * (Q-self.Q_Old)
        self.Q_Old = Q_dash
        self.state_buffer.popleft()
        self.action_buffer.popleft()
        self.reward_buffer.popleft()


        