import rllib.simulators.pythonSimulators as pysim
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

"""config1. Benchmark architecture"""

lr = 1e-3
envName="CartPole-v1"
env = gym.make(envName)
simulator = pysim.GymSimulator(env)

nRuns=2
nEpochs=5000
nTraj=2

class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, simulator.actionSpaceN),
            nn.Softmax(dim=-1))
        
        self.policyNet = [policyNet1]
        
        class ValueNetwork(nn.Module):
            def __init__(self, state_input, action_input, hidden_size):
                super(ValueNetwork, self).__init__()
                self.linearSt = nn.Linear(state_input, hidden_size)
                self.linearAct = nn.Linear(action_input, hidden_size)
                self.linear = nn.Linear(hidden_size, 1)
    
            def forward(self, state, action):
                value = self.linear(F.relu(self.linearSt(state) + \
                                    self.linearAct(action)))
                return value
            
        self.valueNet = nn.Sequential(nn.Linear(simulator.stateSpaceShape[0], 128), nn.ReLU(), nn.Linear(128, 1))
        
        self.valueNetStAct = ValueNetwork(simulator.stateSpaceShape[0], simulator.actionSpaceN+1, 128)
