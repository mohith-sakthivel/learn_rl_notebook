import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam


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
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(
                activ() if i != (len(layer_dims)-2) else out_activ())
        self.mean_net = nn.Sequential(*layers)
        std_log = -0.5 * np.ones(layer_dims[-1], dtype=np.float32)
        self.std_log = nn.Parameter(torch.from_numpy(std_log))

    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.std_log)
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

    def __init__(self, state_dim, act_dim, pol_hid_lyrs=[16],
                 val_hid_lyrs=[16], pol_lr=0.001, val_lr=0.005,
                 batch_size=5000, is_discrete=False, seed=None, use_gpu=True):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.is_discrete = is_discrete
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.pol_lr = pol_lr
        self.val_lr = val_lr
        self.batch_size = batch_size
        # calculate layer dimensions
        pol_layers = [self.state_dim] + pol_hid_lyrs + [self.act_dim]
        val_layers = [self.state_dim] + val_hid_lyrs + [1]
        # instantiate the pytorch nn modules
        if self.is_discrete:
            self.policy = CategoricalPolicy(pol_layers, nn.Tanh, nn.Identity)
        else:
            self.policy = GaussianPolicy(pol_layers, nn.Tanh, nn.Identity)
        self.value_fn = ValueFunction(val_layers, nn.Tanh)
        # set device for computation
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
            self.policy.to(self.device)
            self.value_fn.to(self.device)
        else:
            self.device = torch.device("cpu")
        # setup optimizers
        self.policy_optim = Adam(self.policy.parameters(), self.pol_lr)
        self.value_optim = Adam(self.value_fn.parameters(), self.val_lr)
        # expereience buffers
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

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
        # check if experience buffer is full
        if len(self.reward_buffer) > self.batch_size:
            # Perform policy-value function update
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            policy_loss, value_fn_loss = self.compute_grad(terminal=False)
            policy_loss.backward()
            value_fn_loss.backward()
            self.policy_optim.step()
            self.value_optim.step()
            # Empty buffers
            self.state_buffer = self.state_buffer[-1:]
            self.action_buffer = self.action_buffer[-1:]
            self.reward_buffer.clear()
        return action

    def end(self, reward):
        """
        Agent performs policy update with available experience
        and resets variables for next episode
        """
        self.reward_buffer.append(reward)
        # Perform policy-value function update
        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()
        policy_loss, value_fn_loss = self.compute_grad(terminal=True)
        policy_loss.backward()
        value_fn_loss.backward()
        self.policy_optim.step()
        self.value_optim.step()
        # Empty buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        return policy_loss.item(), value_fn_loss.item()


class REINFORCE(PolicyGradient):
    """
    A policy gradient agent based on REINFORCE algorithm
    """

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
        # calculate returns from rewards
        returns = np.zeros_like(self.reward_buffer, dtype=np.float32)
        returns[-1] = self.reward_buffer[-1]
        if not terminal:
            with torch.no_grad():
                returns[-1] += self.value_fn(state[-1]).item()
        for i in range(len(returns)-2, -1, -1):
            returns[i] = self.reward_buffer[i] + returns[i+1]
        returns = torch.as_tensor(returns,
                                  device=self.device, dtype=torch.float32)
        # calculate value function loss
        state_values = self.value_fn(state[:samples])
        value_fn_loss = nn.functional.mse_loss(state_values, returns)
        # calculate policy loss
        distribution = self.policy(state[:samples])
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action).sum()
        policy_loss = -torch.mean(
                            log_action_probs * (returns-state_values.detach()))
        return policy_loss, value_fn_loss


class ActorCritic(PolicyGradient):
    """
    A policy gradient agent based on Actor_Critic algorithm
    """

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
        returns = torch.tensor(
                              self.reward_buffer,
                              device=self.device, dtype=torch.float32)
        # calculate state values
        state_values = self.value_fn(state)
        with torch.no_grad():
            if not terminal:
                returns[-1] += state_values[-1]
            for i in range(len(returns)-2, -1, -1):
                returns[i] += state_values[i+1]
        # calculate value function loss
        value_fn_loss = nn.functional.mse_loss(state_values[:samples], returns)
        # calculate policy loss
        distribution = self.policy(state[:samples])
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action).sum()
        policy_loss = -torch.mean(
                                  log_action_probs *
                                  (returns-state_values[:samples].detach()))
        return policy_loss, value_fn_loss
