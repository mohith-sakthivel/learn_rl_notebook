import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam


class Critic(nn.Module):
    def __init__(self, layer_dims, activation):
        super().__init__()
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(activation())
        self.value_fn = nn.Sequential(*layers)

    def forward(self, state):
        return self.value_fn(state).squeeze()


class GaussianActor(nn.Module):
    """
    Gaussian Actor for continuous action spaces
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
        self.std_log = nn.Parameter(
                            torch.from_numpy(std_log, device=self.device))

    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.std_log)
        return Normal(mean, std)

    def get_action(self, state):
        with torch.no_grad():
            return self.forward(state).sample().numpy()


class CategoricalActor(nn.Module):
    """
    Categorical Actor for one dimensional discrete action spaces
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


class ActorCritic():
    """
    A policy gradient agent based on Actor-Critic algorithm
    """

    def __init__(self, state_dim, act_dim, act_hid_lyrs=[16],
                 critic_hid_lyrs=[16], actor_lr=0.001, critic_lr=0.005,
                 batch_size=5000, is_discrete=False, seed=None, use_gpu=True):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.is_discrete = is_discrete
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        # calculate layer dimensions
        actor_layers = [self.state_dim] + act_hid_lyrs + [self.act_dim]
        critic_layers = [self.state_dim] + critic_hid_lyrs + [1]
        # instantiate the pytorch nn modules
        if self.is_discrete:
            self.actor = CategoricalActor(actor_layers, nn.Tanh, nn.Identity)
        else:
            self.actor = GaussianActor(actor_layers, nn.Tanh, nn.Identity)
        self.critic = Critic(critic_layers, nn.Tanh)
        # set device for computation
        if torch.cuda.is_available() and use_gpu:
            self.device = torch.device("cuda")
            self.actor.to(self.device)
            self.critic.to(self.device)
        else:
            self.device = torch.device("cpu")
        # setup optimizers
        self.actor_optim = Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)
        # expereience buffers
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def start(self, state):
        """
        Start the agent for the episode
        Recieve the agent state and return an action
        """
        self.state_buffer.append(state)
        state_tensor = torch.as_tensor(
                            state, device=self.device, dtype=torch.float32)
        action = self.actor.get_action(state_tensor)
        self.action_buffer.append(action)
        return action

    def compute_grad(self):
        """
        Computes the loss function for the actor and crtic
        """
        # convert state and action into tensor
        state = torch.tensor(
                self.state_buffer, device=self.device, dtype=torch.float32)
        action = torch.tensor(
                self.action_buffer, device=self.device, dtype=torch.float32)
        # calculate returns from rewards
        returns = np.zeros_like(self.reward_buffer, dtype=np.float32)
        returns[-1] = self.reward_buffer[-1]
        for i in range(len(returns)-2, -1, -1):
            returns[i] = self.reward_buffer[i] + returns[i+1]
        returns = torch.as_tensor(returns,
                                  device=self.device, dtype=torch.float32)
        # calculate critic loss
        state_values = self.critic(state)
        critic_loss = nn.functional.mse_loss(state_values, returns)
        # calculate actor loss
        distribution = self.actor(state)
        if self.is_discrete:
            log_action_probs = distribution.log_prob(action)
        else:
            log_action_probs = distribution.log_prob(action)
        actor_loss = -torch.mean(
            log_action_probs * (returns-state_values.detach()))
        # print('Actor Loss: ', actor_loss.item(),
        #       "\tCritic Loss: ", critic_loss.item())
        return actor_loss, critic_loss

    def take_step(self, reward, state):
        """
        Agent stores experience and selects the next action
        Performs policy update if experience buffer is full
        """
        self.reward_buffer.append(reward)
        state_tensor = torch.as_tensor(
                            state, device=self.device, dtype=torch.float32)
        action = self.actor.get_action(state_tensor)
        # check if experience buffer is full
        if len(self.reward_buffer) > self.batch_size:
            # Perform actor-critic update
            self.reward_buffer[-1] += self.critic(state_tensor).item()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss, critic_loss = self.compute_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            # Empty buffers
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        return action

    def end(self, reward):
        """
        Agent performs policy update with available experience
        and resets variables for next episode
        """
        self.reward_buffer.append(reward)
        # Perform actor-critic update
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        actor_loss, critic_loss = self.compute_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        # Empty buffers
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        return actor_loss.item(), critic_loss.item()
