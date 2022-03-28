import rllib.simulators.pythonSimulators as pysim
import rllib.rlagents.ReinforceA2C as ReinforceA2C
import torch.nn as nn
from gym_minigrid.wrappers import *


lr = 1e-3
envName = "MiniGrid-GoToDoor-5x5-v0"
env = gym.make(envName)
env = FlatObsWrapper(env)
simulator = pysim.GymSimulator(env)

nRuns = 10
nEpochs = 10000
max_step = 800
nTraj = 15
nTrajForGradVar = 50
count_grad_var = 250
entropy_const = 0.075
step_size = 400
gamma = 0.99

baselineLoss = "var"
eval_per_epochs = 100

folderPrefix = "./GoToDoorData/"


def evaluation_Reinforce(current_agent):
    env_eval = gym.make(envName)
    env_eval = FlatObsWrapper(env_eval)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.Reinforce(current_agent.policyNets, simulator_eval, n_trajectories=2)
    
    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats


def evaluation_A2C(current_agent):
    env_eval = gym.make(envName)
    env_eval = FlatObsWrapper(env_eval)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.ReinforceA2CBaseline(current_agent.policyNets, current_agent.valueNet,
                                             simulator_eval, n_trajectories=2)
    
    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats
        
    
def evaluation_EV(current_agent):
    env_eval = gym.make(envName)
    env_eval = FlatObsWrapper(env_eval)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.ReinforceEVBaseline(current_agent.policyNets, current_agent.valueNet,
                                                   simulator_eval, n_trajectories=2, baseline_loss="var")
    
    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats


class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128),
            nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
            nn.Linear(128, simulator.actionSpaceN),
            nn.Softmax(dim=-1))
        
        self.policyNet = [policyNet1]
        
        class ValueNetwork(nn.Module):
            def __init__(self, state_input, action_input, hidden_size):
                super(ValueNetwork, self).__init__()
                self.linearSt = nn.Linear(state_input, hidden_size)
                self.linearAct = nn.Linear(action_input, hidden_size)
                self.act = nn.ReLU()
                self.linear = nn.Linear(hidden_size, 1)
    
            def forward(self, state, action):
                value = self.linear(self.act(self.linearSt(state) + self.linearAct(action)))
                return value
            
        self.valueNet = nn.Sequential(nn.Linear(simulator.stateSpaceShape[0], 128), nn.ReLU(), nn.Linear(128, 1))
        
        self.valueNetStAct = ValueNetwork(simulator.stateSpaceShape[0], simulator.actionSpaceN+1, 128)


