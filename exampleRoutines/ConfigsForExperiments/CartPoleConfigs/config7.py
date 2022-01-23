import rllib.simulators.pythonSimulators as pysim
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

"""config7. Deeper net for baseline"""

lr = 1e-3
envName="CartPole-v1"
env = gym.make(envName)
simulator = pysim.GymSimulator(env)

nRuns=2
nEpochs=5000
nTraj=2

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))

class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128),
            Mish(),
            nn.Linear(128, 128),
            Mish(),
            nn.Linear(128, simulator.actionSpaceN),
            nn.Softmax(dim=-1))
        
        self.policyNet = [policyNet1]
        
        class ValueNetwork(nn.Module):
            def __init__(self, state_input, action_input, hidden_size):
                super(ValueNetwork, self).__init__()
                self.linearSt = nn.Linear(state_input, hidden_size)
                self.linearAct = nn.Linear(action_input, hidden_size)
                self.act = Mish()
                self.linear1 = nn.Linear(hidden_size, 256)
                self.act1 = Mish()
                self.linear2 = nn.Linear(256, 128)
                self.act2 = Mish()
                self.linear3 = nn.Linear(128, 1)
    
            def forward(self, state, action):
                value = self.act(self.linearSt(state) + \
                                    self.linearAct(action))
                value = self.act1(self.linear1(value))
                value = self.act2(self.linear2(value))
                value = self.linear3(value)
                return value
            
        self.valueNet = nn.Sequential(nn.Linear(simulator.stateSpaceShape[0], 128), Mish(), nn.Linear(128, 256), Mish(), nn.Linear(256, 128), Mish(), nn.Linear(128, 1))
        
        self.valueNetStAct = ValueNetwork(simulator.stateSpaceShape[0], simulator.actionSpaceN+1, 128)
