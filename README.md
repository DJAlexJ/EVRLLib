## Structure

The library contains two essential packages: simulators (contains routines for sampling and simulators) and RLAgents (all the implementations of RL Agents are located there).

## Implemented

1.Reinforce
2.Reinforce with state-dependent baselines  
    a. A2C
    b. approaches based on empirical variance minimization

## Requirements

All is runnable on Windows, Linux and Mac OS. There are several packages needed to be installed:
1. PyTorch latest (CPU and GPU version will suffice).
2. Numpy latest.
3. OpenAI Gym if you are planning to work with GymSimulators.
3. Gym-Minigrid if you are planning to work with Minigrid environments (supported by GymSimulators since they have gym-like interface).

## Installation

Download the archive, then being in root folder call
```{bash}
pip install -e .
```

## Usage Examples

All running scripts used for the experiments and config-files are contained in exampleRoutine folder.

