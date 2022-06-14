import rllib.simulators.pythonSimulators as pysim
import rllib.rlagents.ReinforceA2C as ReinforceA2C
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

lr = 1e-3
envName = "Acrobot-v1"
env = gym.make(envName)
simulator = pysim.GymSimulator(env)

nRuns = 1
nEpochs = 5000
max_step = 500
nTraj = 2
nTrajForGradVar = 50
count_grad_var = 200
entropy_const = 0.075
step_size = 750
gamma = 0.99

baselineLoss = "var"
eval_per_epochs = 200

folderPrefix = "./data25042022/"


def evaluation_Reinforce(current_agent):
    env_eval = gym.make(envName)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.Reinforce(current_agent.policyNets,
                                        simulator_eval, n_trajectories=2)

    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats


def evaluation_A2C(current_agent):
    env_eval = gym.make(envName)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.ReinforceA2CBaseline(current_agent.policyNets,
                                                   current_agent.valueNet,
                                                   simulator_eval,
                                                   n_trajectories=2)

    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats


def evaluation_EV(current_agent):
    env_eval = gym.make(envName)
    simulator_eval = pysim.GymSimulator(env_eval)
    agent_eval = ReinforceA2C.ReinforceEVBaseline(current_agent.policyNets,
                                                  current_agent.valueNet,
                                                  simulator_eval,
                                                  n_trajectories=2,
                                                  baseline_loss="var")

    stats = agent_eval.evaluate(n_samples=5, max_step=3000)
    return stats


class PolicyValuePair:
    def __init__(self):
        policyNet1 = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128),
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
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, state, action):
                value = self.linear(
                    self.act(self.linearSt(state) + self.linearAct(action)))
                return value

        self.valueNet = nn.Sequential(
            nn.Linear(simulator.stateSpaceShape[0], 128), nn.ReLU(),
            nn.Linear(128, 1))

        self.valueNetStAct = ValueNetwork(simulator.stateSpaceShape[0],
                                          simulator.actionSpaceN + 1, 128)
