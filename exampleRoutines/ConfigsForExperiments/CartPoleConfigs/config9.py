import rllib.simulators.pythonSimulators as pysim
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

"""config9. Same as config8, but with ReLU and learning_rate = 3e-4"""

lr = 3e-4
envName="CartPole-v1"
env = gym.make(envName)
simulator = pysim.GymSimulator(env)

nRuns=2
nEpochs=5000
nTraj=2

class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, simulator.actionSpaceN),
            nn.Softmax(dim=-1))
        
        self.policyNet = [policyNet1]
        
        class ValueNetwork(nn.Module):
            def __init__(self, state_input, action_input, hidden_size):
                super(ValueNetwork, self).__init__()
                self.linearSt = nn.Linear(state_input, hidden_size)
                self.linearAct = nn.Linear(action_input, hidden_size)
                self.act = nn.ReLU()
                self.linear1 = nn.Linear(hidden_size, 256)
                self.act1 = nn.ReLU()
                self.linear2 = nn.Linear(256, 128)
                self.act2 = nn.ReLU()
                self.linear3 = nn.Linear(128, 1)
    
            def forward(self, state, action):
                value = self.act(self.linearSt(state) + \
                                    self.linearAct(action))
                value = self.act1(self.linear1(value))
                value = self.act2(self.linear2(value))
                value = self.linear3(value)
                return value
            
        self.valueNet = nn.Sequential(nn.Linear(simulator.stateSpaceShape[0], 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        
        self.valueNetStAct = ValueNetwork(simulator.stateSpaceShape[0], simulator.actionSpaceN+1, 128)
