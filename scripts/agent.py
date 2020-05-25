import numpy as np
from collections import deque
import tiles3


class TileEncoder():
    def __init__(self, var_ranges, num_tiles, num_tilings):
        assert len(var_ranges) == len(num_tiles), \
            "Input variables length do not match"
        assert all(isinstance(val, int) and val > 0 for val in num_tiles), \
            "number of tiles should be an array of integers > 0" 
        assert all(len(var_range) == 2 for var_range in var_ranges) , \
            "variable range should be a finite numeric interval"
        assert isinstance(num_tilings, int), \
            "number of tilings should be an integer" 
        self.var_ranges = var_ranges
        self.num_tiles = num_tiles
        self.num_var = len(var_ranges)
        self.num_tilings = num_tilings
        self.var_coeff = np.zeros(self.num_var)
        self.get_coeffs()
        self.iht_size = self.calc_iht_size()
        self.iht = tiles3.IHT(self.iht_size)

    def get_coeffs(self):
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
        new_values = np.array(values) * self.var_coeff
        return tiles3.tiles(self.iht, self.num_tilings, new_values)


class SARSA_agent():
    """
    n-step semi gradient SARSA agent for episodic tasks with discrete action 
    space and continuous state space
    """

    def __init__(self, agent_info):
        self.tc = TileEncoder(
                              agent_info.get("obs_limits"),
                              agent_info.get("num_tiles"),
                              agent_info.get("num_tilings"))
        self.policy_rand_generator = np.random.default_rng(agent_info.get("seed"))
        self.n_step = agent_info.get("n_step")
        self.num_actions = agent_info.get("num_actions")
        self.epsilon = agent_info.get("epsilon")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")/self.tc.num_tilings
        self.d_step_size = self.step_size
        self.value_function = np.zeros((self.num_actions, self.tc.iht_size, self.tc.num_tilings))
        self.last_state = deque()
        self.last_action = deque()
        self.last_rewards = deque()
        self.time_step = 0
        self.disc_powers = [np.power(self.discount, i) for i in range(self.n_step)]
        self.disc_n_power = np.power(self.discount, self.n_step)

    def agent_policy(self, state):
        """
        Return an action according to the policy given a state
        """
        if self.policy_rand_generator.random() < self.epsilon:
            return self.policy_rand_generator.integers(0, self.num_actions)
        else:
            indices = [i for i in range(self.tc.num_tilings)]
            action_vals= np.sum(self.value_function[:, self.tc.get_feature(state), indices], axis=1)
            max_val = np.amax(action_vals)
            max_actions = np.where(action_vals == max_val)[0]
            return self.policy_rand_generator.choice(max_actions)

    def start(self, state):
        """ Start the agent for the episode """
        self.last_state.append(state)
        self.last_action.append(self.agent_policy(state))
        return self.last_action[0]

    def take_step(self, reward, state):
        """
        Agent updates the state values and returns the next action
        """
        self.time_step += 1
        action = self.agent_policy(state)
        self.last_state.append(state)
        self.last_action.append(action)
        self.last_rewards.append(reward)

        update_timestep = self.time_step - self.n_step
        if update_timestep >= 0:
            sampled_returns = 0
            for i in range(self.n_step):
                sampled_returns += self.disc_powers[i] * self.last_rewards[i]

            update_state_fv = self.tc.get_feature(self.last_state[0])
            fv = self.tc.get_feature(state)
            indices = [i for i in range(self.tc.num_tilings)]
            target = sampled_returns + self.disc_n_power * np.sum(self.value_function[action, fv, indices])
            
            td_error = target - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
            self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
            self.last_state.popleft()
            self.last_action.popleft()
            self.last_rewards.popleft()
        return action

    def end(self, reward):
        """
        Terminate the episode and update the values
        """
        self.time_step += 1
        self.last_rewards.append(reward)

        while self.last_rewards:
            num_bootsrap = len(self.last_rewards)
            sampled_returns = 0
            for i in range(num_bootsrap):
                sampled_returns += self.disc_powers[i] * self.last_rewards[i]

            update_state_fv = self.tc.get_feature(self.last_state[0])
            indices = [i for i in range(self.tc.num_tilings)]
            
            td_error = sampled_returns - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
            self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
            self.last_state.popleft()
            self.last_action.popleft()
            self.last_rewards.popleft()
        self.reset_episode()

    def reset_episode(self):
        """ Resets episode related parameters of the agent """
        self.last_state.clear()
        self.last_action.clear()
        self.last_rewards.clear()
        self.time_step = 0


class Expected_SARSA_agent():
    """
    n-step expected semi gradient SARSA agent for episodic tasks with discrete
    action space and continuous state space
    """

    def __init__(self, agent_info):
        self.tc = TileEncoder(
                              agent_info.get("obs_limits"),
                              agent_info.get("num_tiles"),
                              agent_info.get("num_tilings"))
        self.policy_rand_generator = np.random.default_rng(agent_info.get("seed"))
        self.n_step = agent_info.get("n_step")
        self.num_actions = agent_info.get("num_actions")
        self.epsilon = agent_info.get("epsilon")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")/self.tc.num_tilings
        self.d_step_size = self.step_size
        self.value_function = np.zeros((self.num_actions, self.tc.iht_size, self.tc.num_tilings))
        self.last_state = deque()
        self.last_action = deque()
        self.last_rewards = deque()
        self.time_step = 0
        self.disc_powers = [np.power(self.discount, i) for i in range(self.n_step)]
        self.disc_n_power = np.power(self.discount, self.n_step)

    def agent_policy(self, state):
        """
        Return an action according to the policy given a state
        """
        if self.policy_rand_generator.random() < self.epsilon:
            return self.policy_rand_generator.integers(0, self.num_actions)
        else:
            indices = [i for i in range(self.tc.num_tilings)]
            action_vals= np.sum(self.value_function[:, self.tc.get_feature(state), indices], axis=1)
            max_val = np.amax(action_vals)
            max_actions = np.where(action_vals == max_val)[0]
            return self.policy_rand_generator.choice(max_actions)

    def start(self, state):
        """ Start the agent for the episode """
        self.last_state.append(state)
        self.last_action.append(self.agent_policy(state))
        return self.last_action[0]

    def take_step(self, reward, state):
        """
        Agent updates the state values and returns the next action
        """
        self.time_step += 1
        action = self.agent_policy(state)
        self.last_state.append(state)
        self.last_action.append(action)
        self.last_rewards.append(reward)

        update_timestep = self.time_step - self.n_step
        if update_timestep >= 0:
            sampled_returns = 0
            for i in range(self.n_step):
                sampled_returns += self.disc_powers[i] * self.last_rewards[i]

            update_state_fv = self.tc.get_feature(self.last_state[0])
            fv = self.tc.get_feature(state)
            indices = [i for i in range(self.tc.num_tilings)]
            weights = np.zeros(self.num_actions)
            action_values = np.sum(self.value_function[:, fv, indices], axis=1)
            max_val = np.amax(action_values)
            max_actions = np.where(action_values == max_val)[0]
            weights[max_actions] = (1-self.epsilon)/max_actions.size
            weights += self.epsilon/self.num_actions
            target = sampled_returns + self.disc_n_power * np.sum(weights * action_values)
            
            td_error = target - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
            self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
            self.last_state.popleft()
            self.last_action.popleft()
            self.last_rewards.popleft()
        return action

    def end(self, reward):
        """
        Terminate the episode and update the values
        """
        self.time_step += 1
        self.last_rewards.append(reward)

        while self.last_rewards:
            num_bootsrap = len(self.last_rewards)
            sampled_returns = 0
            for i in range(num_bootsrap):
                sampled_returns += self.disc_powers[i] * self.last_rewards[i]

            update_state_fv = self.tc.get_feature(self.last_state[0])
            indices = [i for i in range(self.tc.num_tilings)]
            
            td_error = sampled_returns - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
            self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
            self.last_state.popleft()
            self.last_action.popleft()
            self.last_rewards.popleft()
        self.reset_episode()

    def reset_episode(self):
        """ Resets episode related parameters of the agent """
        self.last_state.clear()
        self.last_action.clear()
        self.last_rewards.clear()
        self.time_step = 0

class Q_agent():
    """
    Q-Learning agent for episodic tasks with discrete
    action space and continuous state space
    """

    def __init__(self, agent_info):
        self.tc = TileEncoder(
                              agent_info.get("obs_limits"),
                              agent_info.get("num_tiles"),
                              agent_info.get("num_tilings"))
        self.policy_rand_generator = np.random.default_rng(agent_info.get("seed"))
        self.num_actions = agent_info.get("num_actions")
        self.epsilon = agent_info.get("epsilon")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")/self.tc.num_tilings
        self.d_step_size = self.step_size
        self.value_function = np.zeros((self.num_actions, self.tc.iht_size, self.tc.num_tilings))
        self.last_state = deque()
        self.last_action = deque()
        self.last_rewards = deque()
        self.time_step = 0

    def agent_policy(self, state):
        """
        Return an action according to the policy given a state
        """
        if self.policy_rand_generator.random() < self.epsilon:
            return self.policy_rand_generator.integers(0, self.num_actions)
        else:
            indices = [i for i in range(self.tc.num_tilings)]
            action_vals= np.sum(self.value_function[:, self.tc.get_feature(state), indices], axis=1)
            max_val = np.amax(action_vals)
            max_actions = np.where(action_vals == max_val)[0]
            return self.policy_rand_generator.choice(max_actions)

    def start(self, state):
        """ Start the agent for the episode """
        self.last_state.append(state)
        self.last_action.append(self.agent_policy(state))
        return self.last_action[0]

    def take_step(self, reward, state):
        """
        Agent updates the state values and returns the next action
        """
        self.time_step += 1
        action = self.agent_policy(state)
        self.last_state.append(state)
        self.last_action.append(action)
        self.last_rewards.append(reward)


        update_state_fv = self.tc.get_feature(self.last_state[0])
        fv = self.tc.get_feature(state)
        indices = [i for i in range(self.tc.num_tilings)]
        action_value = np.amax(np.sum(self.value_function[:, fv, indices], axis=1))

        target = reward + self.discount * action_value
            
        td_error = target - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
        self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
        self.last_state.popleft()
        self.last_action.popleft()
        self.last_rewards.popleft()
        return action

    def end(self, reward):
        """
        Terminate the episode and update the values
        """
        self.time_step += 1
        self.last_rewards.append(reward)

        update_state_fv = self.tc.get_feature(self.last_state[0])
        indices = [i for i in range(self.tc.num_tilings)]
            
        td_error = reward - np.sum(self.value_function[self.last_action[0], update_state_fv, indices])
        self.value_function[self.last_action[0], update_state_fv, indices] += self.d_step_size * td_error
        self.last_state.popleft()
        self.last_action.popleft()
        self.last_rewards.popleft()
        self.reset_episode()

    def reset_episode(self):
        """ Resets episode related parameters of the agent """
        self.last_state.clear()
        self.last_action.clear()
        self.last_rewards.clear()
        self.time_step = 0