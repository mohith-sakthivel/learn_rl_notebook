
### Optimal Agent for MountainCar-v0
A highly optimized RL agent is presented for the MountainCar-v0 problem.

Refer the notebook for the implementation

#### Hyperparameters:

    - Algorithm: True Online SARSA (lambda) algorithm
    - Step Size - 0.1 (Step size decays exponentially with a factor of 0.99 every timestep)
    - lambda - 0.9
    - epsilon - 0.001
    - Tile Encoder - 16 Tilings with 8 tiles per state space dimension
    - Discount - 1


