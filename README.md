# Learn RL Notebook

## Introduction
This repository contains implementations of various reinforcement learning algorithms.

It also has notebooks that walk you through those algorithms comparing them and briefing the effects of their hyper-parameters.

#### Note: Check out the images folder for pre-generated graphs and visualizations to get an idea of the kind of intuition the notebook offers.

### Example Enviroments
The repository contains examples implemented on the following environments

    - MountainCar-v01
    - MountainCarContinuous-v0
    - CartPole-v0

## Available Agents

### Classical RL
    - Semi-Gradient n-step SARSA agent
    - Semi-Gradient n-step Expected SARSA agent
    - Semi-Gradient Q-Learning agent
    - True Online SARSA(lambda)

### Deep RL 
    - REINFORCE
    - Actor-Critic


## Notebook Descriptions

## 1. classic_rl_notebook:
### Description
This notebook was developed to help people get started on implementing Reinforcement Learning algorithms.

This notebook could be a nice supplement to someone reading 'Reinforcement Learning - An Introduction' by Richard Sutton and Andrew Barto.

The goal of the notebook is to

- Help enthusiasts understand the effect of various hyperparameters on agent performance
- Compare various RL algorithms

The MountainCar-v0 problem is used to illustrate various key insights.

### Optimal Agent for MountainCar-v0 (openAI lederboard)
A highly optimized RL agent is presented for the MountainCar-v0 problem.

Refer the last cell of the classic_rl_notebook (in the scripts folder)

#### Hyperparameters:

    - Algorithm: True Online SARSA (lambda) algorithm
    - Step Size - 0.1 (Step size decays exponentially with a factor of 0.99 every timestep)
    - lambda - 0.9
    - epsilon - 0.001
    - Tile Encoder - 16 Tilings with 8 tiles per state space dimension
    - Discount - 1
    

#### Note: For further study on eligibility traces, turn to the Chapter-12 of Prof. Sutton's book.


