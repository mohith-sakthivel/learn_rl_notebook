# Learn RL Notebook

## Introduction
This repository contains implementations of various reinforcement learning algorithms.

It also has notebooks that walk you through those algorithms comparing them and briefing the effects of their hyper-parameters.

#### Note: Check out the images folder for pre-generated graphs and visualizations to get an idea of the kind of intuition the notebook offers.

### Example Enviroments
The repository contains examples implemented on the following environments

    - MountainCar-v0
    - CartPole-v1

## Available Agents

### Classical RL
    - Semi-Gradient n-step SARSA agent
    - Semi-Gradient n-step Expected SARSA agent
    - Semi-Gradient Q-Learning agent
    - True Online SARSA(lambda)

### Deep RL 
    - REINFORCE
    - REINFORCE with Baseline
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


## 2. policy_gradient_notebook:
### Description

Policy gradient algorithms are a class of RL algorithms that are becoming very popular. They are widely adopted for multiple continuous control tasks like Robotics and Autonomous Driving.

This notebook contains implementaions of some very popular basic policy gradients algorithms.

Specifically the notebook compares the peformance of the following three algorithms and provides some insights.
    - REINFORCE
    - REINFORCE with Baseline
    - Actor-Critic

The example used in this notebook is the CartPole-v1 problem.