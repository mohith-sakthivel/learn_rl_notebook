import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class ValueFunction(nn.Module):
    def __init__(self, layer_dims, activation):
        super().__init__()
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(activation())
        self.value_fn = nn.Sequential(*layers)

    def forward(self, state):
        return self.value_fn(state).squeeze(dim=-1)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces
    """
    def __init__(self, layer_dims, activ, out_activ):
        super().__init__()
        mean_layers = [None] * (2 * (len(layer_dims)-1))
        std_layers = [None] * (2 * (len(layer_dims)-1))
        for i in range(len(layer_dims)-1):
            mean_layers[2*i] = nn.Linear(layer_dims[i], layer_dims[i+1])
            if i < (len(layer_dims)-1)//2:
                std_layers[2*i] = mean_layers[2*i]
            else:
                std_layers[2*i] = nn.Linear(layer_dims[i], layer_dims[i+1])
            mean_layers[(2*i)+1] = std_layers[(2*i)+1] = \
                activ() if i != (len(layer_dims)-2) else out_activ()
        self.mean_net = nn.Sequential(*mean_layers)
        self.log_std_net = nn.Sequential(*std_layers)

        # layers = []
        # for i in range(len(layer_dims)-1):
        #     layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        #     layers.append(
        #         activ() if i != (len(layer_dims)-2) else out_activ())
        # self.mean_net = nn.Sequential(*layers)
        # std_log = -0.5 * np.ones(layer_dims[-1], dtype=np.float32)
        # self.std_log = nn.Parameter(torch.from_numpy(std_log))

    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std_net(state))
        # std = torch.exp(self.std_log)
        return Normal(mean, std)

    def get_action(self, state):
        with torch.no_grad():
            return self.forward(state).sample().tolist()


class CategoricalPolicy(nn.Module):
    """
    Categorical policy for one dimensional discrete action spaces
    """

    def __init__(self, layer_dims, activ, out_activ):
        super().__init__()
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(
                activ() if i != (len(layer_dims)-2) else out_activ())
        self.logit_net = nn.Sequential(*layers)

    def forward(self, state):
        return Categorical(logits=self.logit_net(state))

    def get_action(self, state):
        with torch.no_grad():
            return self.forward(state).sample().item()


class PolicyGradient():
    """
    A policy gradient agent implementation
    """

    def __init__(self, state_dim, act_dim,
                 pol_hid_lyrs=[16], val_hid_lyrs=[16], pol_lr=0.001,
                 val_lr=0.005, lr_decay=1, min_pol_lr=0, min_val_lr=0,
                 pol_act=nn.Tanh, val_act=nn.Tanh, discount=1, batch_size=5000,
                 is_discrete=False, baseline=True, seed=None, use_gpu=True):
        # Seed works properly only when you run on cpu
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.is_discrete = is_discrete
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.pol_lr = pol_lr
        self.lr_decay = lr_decay
        self.min_pol_lr = min_pol_lr
        self.discount = discount
        self.batch_size = batch_size
        self.rewards_processed = 0
        self.baseline = baseline
        # set device for computation
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Instantiate policy neural network
        pol_layers = [self.state_dim] + pol_hid_lyrs + [self.act_dim]
        if self.is_discrete:
            self.policy = CategoricalPolicy(pol_layers, pol_act, nn.Identity)
        else:
            self.policy = GaussianPolicy(pol_layers, pol_act, nn.Identity)
        self.policy.to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), self.pol_lr)
        self.policy_lr_sch = ExponentialLR(self.policy_optim, self.lr_decay)
        self.policy_loss = torch.zeros(1, device=self.device,
                                       dtype=torch.float32)
        # Instantiate neural network for value function
        if self.baseline:
            val_layers = [self.state_dim] + val_hid_lyrs + [1]
            self.val_lr = val_lr
            self.min_val_lr = min_val_lr
            self.value_fn = ValueFunction(val_layers, val_act)
            self.value_fn.to(self.device)
            self.value_optim = Adam(self.value_fn.parameters(), self.val_lr)
            self.value_lr_sch = ExponentialLR(self.value_optim, self.lr_decay)
            self.value_fn_loss = torch.zeros(1, device=self.device,
                                             dtype=torch.float32)
        # expereience buffers
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.discount_buffer = []

    def compute_grad(self, terminal=True):
        """ Calculate the gradient of the policy and value fn """
        raise NotImplementedError

    def start(self, state):
        """
        Start the agent for the episode
        Recieve the agent state and return an action
        """
        self.state_buffer.append(state)
        state_tensor = torch.as_tensor(
                            state, device=self.device, dtype=torch.float32)
        action = self.policy.get_action(state_tensor)
        self.action_buffer.append(action)
        self.episode_reward = 0
        self.discount_buffer.append(1)
        return action

    def take_step(self, reward, state):
        """
        Agent stores experience and selects the next action
        Performs policy update if experience buffer is full
        """
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
        state_tensor = torch.as_tensor(
                            state, device=self.device, dtype=torch.float32)
        action = self.policy.get_action(state_tensor)
        self.action_buffer.append(action)
        self.discount_buffer.append(self.discount_buffer[-1] * self.discount)
        self.episode_reward += reward
        return action

    def end(self, reward):
        """
        Agent performs policy update with available experience
        and resets variables for next episode
        """
        self.reward_buffer.append(reward)
        self.episode_reward += reward
        # Perform policy update
        policy_loss, value_fn_loss = self.compute_grad(True)
        # Empty buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.discount_buffer.clear()
        return self.episode_reward, policy_loss, value_fn_loss


class REINFORCE(PolicyGradient):
    """
    A policy gradient agent based on REINFORCE algorithm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, baseline=False, **kwargs)

    def __str__(self):
        return "REINFORCE"

    def compute_grad(self, terminal):
        """
        Computes the loss function for the policy and gradient
        """
        # convert state and action into tensor
        samples = len(self.reward_buffer)
        state = torch.tensor(
                             self.state_buffer,
                             device=self.device, dtype=torch.float32)
        action = torch.tensor(
                              self.action_buffer[:samples],
                              device=self.device, dtype=torch.float32)
        discount = torch.tensor(
                                self.discount_buffer[:samples],
                                device=self.device, dtype=torch.float32)
        returns = torch.zeros(len(self.reward_buffer), device=self.device,
                              dtype=torch.float32)
        # calculate returns from rewards
        returns[-1] = self.reward_buffer[-1]
        for i in range(len(returns)-2, -1, -1):
            returns[i] = self.reward_buffer[i] + self.discount * returns[i+1]
        # calculate policy loss
        distribution = self.policy(state[:samples])
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action).sum(dim=-1)
        # Calculate loss and backpropagate
        self.policy_loss += -torch.sum(log_action_probs * (discount * returns))
        # Update parameters
        if self.rewards_processed + len(self.reward_buffer) >= self.batch_size:
            self.policy_loss = self.policy_loss / (self.rewards_processed +
                                                   len(self.reward_buffer))
            self.policy_loss.backward()
            self.policy_optim.step()
            if self.policy_lr_sch.get_last_lr()[0] > self.min_pol_lr:
                self.policy_lr_sch.step()
            self.policy_optim.zero_grad()
            self.rewards_processed = 0
            policy_loss = self.policy_loss.item()
            self.policy_loss = torch.zeros(1, device=self.device,
                                           dtype=torch.float32)
            return policy_loss, 0
        else:
            self.rewards_processed += len(self.reward_buffer)
            return None, None


class REINFORCE_Baseline(PolicyGradient):
    """
    A policy gradient agent based on REINFORCE algorithm with baseline
    """
    def __str__(self):
        return "REINFORCE_Baseline"

    def compute_grad(self, terminal):
        """
        Computes the loss function for the policy and gradient
        """
        # convert state and action into tensor
        samples = len(self.reward_buffer)
        state = torch.tensor(
                             self.state_buffer,
                             device=self.device, dtype=torch.float32)
        action = torch.tensor(
                              self.action_buffer[:samples],
                              device=self.device, dtype=torch.float32)
        discount = torch.tensor(
                                self.discount_buffer[:samples],
                                device=self.device, dtype=torch.float32)
        returns = torch.zeros(len(self.reward_buffer), device=self.device,
                              dtype=torch.float32)
        # calculate value function
        state_values = self.value_fn(state[:samples])
        # calculate returns from rewards
        with torch.no_grad():
            returns[-1] = self.reward_buffer[-1]
            # Code to handle non-termial cases
            # if not terminal:
            #     returns[-1] += self.discount*self.value_fn(state[-1]).item()
            for i in range(len(returns)-2, -1, -1):
                returns[i] = self.reward_buffer[i] + self.discount*returns[i+1]
            # calculate TD error
            td_error = returns - state_values
        # calculate value loss
        self.value_fn_loss += nn.functional.mse_loss(state_values, returns,
                                                     reduction='sum')
        # calculate policy loss
        distribution = self.policy(state[:samples])
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action).sum(dim=-1)
        self.policy_loss += -torch.sum(discount * td_error * log_action_probs)
        # Update parameters
        if self.rewards_processed + len(self.reward_buffer) >= self.batch_size:
            # Calculate mean loss
            self.policy_loss = self.policy_loss / (self.rewards_processed +
                                                   len(self.reward_buffer))
            self.value_fn_loss = self.value_fn_loss / (self.rewards_processed +
                                                       len(self.reward_buffer))
            # Backpropagate
            self.policy_loss.backward()
            self.value_fn_loss.backward()
            # Update parameters
            self.policy_optim.step()
            self.value_optim.step()
            # Schedule learning rate
            if self.policy_lr_sch.get_last_lr()[0] > self.min_pol_lr:
                self.policy_lr_sch.step()
            if self.value_lr_sch.get_last_lr()[0] > self.min_val_lr:
                self.value_lr_sch.step()
            # Clear gradients
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            # Reset variables
            policy_loss = self.policy_loss.item()
            value_fn_loss = self.value_fn_loss.item()
            self.rewards_processed = 0
            self.policy_loss = torch.zeros(1, device=self.device,
                                           dtype=torch.float32)
            self.value_fn_loss = torch.zeros(1, device=self.device,
                                             dtype=torch.float32)
            return policy_loss, value_fn_loss
        else:
            self.rewards_processed += len(self.reward_buffer)
            return None, None


class ActorCritic(PolicyGradient):
    """
    A policy gradient agent based on Actor_Critic algorithm
    """
    def __str__(self):
        return "Actor-Critic"

    def compute_grad(self, terminal=True):
        """
        Computes the loss function for the policy and gradient
        """
        # convert state and action into tensor
        samples = len(self.reward_buffer)
        state = torch.tensor(
                             self.state_buffer,
                             device=self.device, dtype=torch.float32)
        action = torch.tensor(
                              self.action_buffer[:samples],
                              device=self.device, dtype=torch.float32)
        discount = torch.tensor(
                                self.discount_buffer[:samples],
                                device=self.device, dtype=torch.float32)
        returns = torch.tensor(
                              self.reward_buffer,
                              device=self.device, dtype=torch.float32)
        # calculate value function
        state_values = self.value_fn(state[:samples])
        # calculate returns from rewards
        with torch.no_grad():
            returns[:-1] += self.discount * state_values[1:samples]
            # Code to handle non-terminal cases
            # if not terminal:
            #     returns[-1] += self.discount*self.value_fn(state[-1]).item()

            # calculate TD error
            td_error = returns - state_values
        self.value_fn_loss += nn.functional.mse_loss(state_values, returns,
                                                     reduction='sum')
        # calculate policy loss
        distribution = self.policy(state[:samples])
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action).sum(dim=-1)
        self.policy_loss += -torch.sum(discount * td_error * log_action_probs)
        # Update parameters
        if self.rewards_processed + len(self.reward_buffer) >= self.batch_size:
            # Calculate mean loss
            self.policy_loss = self.policy_loss / (self.rewards_processed +
                                                   len(self.reward_buffer))
            self.value_fn_loss = self.value_fn_loss / (self.rewards_processed +
                                                       len(self.reward_buffer))
            # Backpropagate
            self.policy_loss.backward()
            self.value_fn_loss.backward()
            # Update parameters
            self.policy_optim.step()
            self.value_optim.step()
            # Schedule learning rate
            if self.policy_lr_sch.get_last_lr()[0] > self.min_pol_lr:
                self.policy_lr_sch.step()
            if self.value_lr_sch.get_last_lr()[0] > self.min_val_lr:
                self.value_lr_sch.step()
            # Clear gradients
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            # Reset variables
            policy_loss = self.policy_loss.item()
            value_fn_loss = self.value_fn_loss.item()
            self.rewards_processed = 0
            self.policy_loss = torch.zeros(1, device=self.device,
                                           dtype=torch.float32)
            self.value_fn_loss = torch.zeros(1, device=self.device,
                                             dtype=torch.float32)
            return policy_loss, value_fn_loss
        else:
            self.rewards_processed += len(self.reward_buffer)
            return None, None
