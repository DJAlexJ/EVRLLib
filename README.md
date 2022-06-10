## Structure

The library contains two packages: simulators (contains routines for sampling trajectories from environments and simulators) and RLAgents (all the implementations of RL Agents are located there).

## Implemented

1. Reinforce
2. Reinforce with state-dependent baselines  
    1. A2C
    2. EV (Empirical Variance Minimization)
        1. EVv (variance)
        2. EVm (2nd moment)

## Requirements
```
torch >= 1.7.1
numpy >= 1.17.2
gym >= 0.18.0
gym-minigrid >= 1.0.2 if you are planning to work with Minigrid environments (supported by GymSimulators since they have gym-like interface)
```

## Installation

Download the archive, then being in root folder call
```bash
pip install -e .
```

## Agents params
Agent's init() method has following arguments:
1. list_policy_net: List[torch.nn.Module] -- list of policies nets, modelling the policy
2. value_net: torch.nn.Module -- value network
3. simulator: simulator -- simulator object
4. n_trajectories: int -- number of trajectories in MC estimate of the gradient
5. policy: str -- distribution to sample actions for continuous env
6. device: str -- device to use with torch
7. baseline_loss: str -- loss to use for baseline training (ONLY IF EVM USED)
    var: full empirical variance
    2ndMoment: only the second moment
8. nTrajectoriesForGradVar: int -- number of trajectories to evaluate gradient variance

Agent's train() method has following arguments:
1. n_epochs: int -- number of epochs
2. max_step: int -- maximum length of sampled trajectory from the simulator
3. lr: float -- learning rate, parameter of the optimizer
4. eval_func: func -- function for evaluation: accepts current agent and return statistics after agent.evaluate(n_samples, max_step)
5. eval_per_epochs: int -- perform eval_func each eval_per_epochs epochs
6. step_size: int -- scheduler stepsize, parameter of the optimizer
7. gamma: float -- discounting factor
8. entropy_const: float -- const by which the policy entropy is multiplied
9. verbose: int -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
10. count_grad_variance: int -- count gradient variance each count_grad_variance epochs. -1 value disables this option


## Usage Examples

Here is a simple example, how to use simulator and RLagent for training in gym environment:
```python
# import main libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdirllib.simulators import pythonSimulators as pySim
import hdirllib.rlagents.ReinforceA2C as ReinforceA2C

import time
import pickle
import gym

import matplotlib.pyplot as plt


env_name = "CartPole-v1"  # choose environment
env = gym.make(env_name)
simulator = pySim.GymSimulator(env)  # initialize simulator

# create simple policy and simple baseline
class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128),
            nn.Linear(128, simulator.actionSpaceN),
            nn.Softmax(dim=-1)
        )
        
        self.policyNet = [policyNet1]
        self.valueNet = nn.Sequential(nn.Linear(simulator.stateSpaceShape[0], 128), nn.ReLU(), nn.Linear(128, 1))
        
        
polval = PolicyValuePair()
policyNets = polval.policyNet
valueNet = polval.valueNet

# initialize A2C agent
agent = ReinforceA2C.ReinforceA2CBaseline(
    policyNets, 
    valueNet, 
    simulator, 
    n_trajectories=2, 
    nTrajectoriesForGradVar=2
)

# function for intermediate evaluation (not necessary)
def evaluation_A2C(current_agent):
    env_eval = gym.make(env_name)
    simulator_eval = pySim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.ReinforceA2CBaseline(current_agent.policyNets, current_agent.valueNet,
                                                   simulator_eval, n_trajectories=2)
    
    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats



logsA2C = agent.train(
    n_epochs=600, 
    lr=1e-3, 
    max_step=200, 
    verbose=50, 
    entropy_const=0.075,
    step_size=750, 
    gamma=0.99,
    eval_func=evaluation_A2C, 
    eval_per_epochs=100, 
    count_grad_variance=-1
)
```
After training is completed, logsA2C contain different statistics. For example:
```python
plt.figure(figsize=(13, 7))
plt.grid()
plt.ylabel("Reward")
plt.xlabel("Epoch")
plt.plot(pd.Series(logsA2C['meanRewards']).rolling(10).mean())
```

<img width="789" alt="image" src="https://user-images.githubusercontent.com/18465332/172884353-7d5b4eb0-4fbb-4a31-a7f9-7c0984277308.png">
