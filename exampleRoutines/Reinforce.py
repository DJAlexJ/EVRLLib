import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import rllib.rlagents.ReinforceA2C as srlagents
import rllib.simulators.pythonSimulators as pysim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import importlib
import time
import tqdm
import pickle

import os
import gym
from gym_minigrid.wrappers import *

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, required=True)

args = parser.parse_args()

config = importlib.import_module(args.config_path)
config_name = args.config_path.rsplit('.')[-1]

#set environment
envName = config.envName
env = config.env
simulator = config.simulator

nRuns = config.nRuns
nEpochs = config.nEpochs
nTraj = config.nTraj
nTrajForGradVar = config.nTrajForGradVar
gamma = config.gamma
step_size = config.step_size
entropy_const = config.entropy_const
max_step = config.max_step
lr = config.lr
eval_per_epochs = config.eval_per_epochs
count_grad_var = config.count_grad_var

evaluation_Reinforce = config.evaluation_Reinforce
    
folderPrefix=config.folderPrefix
nameAppendix='lr'+str(config.lr)+'_'+str(np.random.randint(10050000))

logsReinf=np.zeros([nRuns,nEpochs])
evalReinf=np.zeros([nRuns,int(np.ceil(nEpochs/eval_per_epochs))])
timesReinf=np.zeros([nRuns])
trajLensReinf=np.zeros([nRuns,nEpochs]) # np.mean(trajLens) is returned so multiply by nTraj
gradVarReinf=np.zeros([nRuns, int(np.ceil(nEpochs/count_grad_var))])
gradVarRewards=np.zeros([nRuns, int(np.ceil(nEpochs/count_grad_var))])

for runId in np.arange(nRuns):
    print("RUN ",runId)
    
    polval = config.PolicyValuePair()
    policyNets = polval.policyNet
    valueNet = polval.valueNet
    agent = srlagents.Reinforce(policyNets, simulator, n_trajectories=nTraj, nTrajectoriesForGradVar=nTrajForGradVar)

    time0 = time.time()
    logs = agent.train(n_epochs=nEpochs, lr=lr, max_step=max_step, eval_func=evaluation_Reinforce,
                       step_size=step_size, gamma=gamma, entropy_const=entropy_const,
                       eval_per_epochs=eval_per_epochs, count_grad_variance=count_grad_var)
    timesReinf[runId] = time.time()-time0
    logsReinf[runId,:] = logs['meanRewards']
    evalReinf[runId,:] = list(map(lambda x: x['rewardMean'], logs['evalInfo']))
    trajLensReinf[runId,:] = logs['nSteps']*nTraj
    gradVarReinf[runId,:] = logs["GradientVariance"]
    gradVarRewards[runId,:] = logs["meanRewardsGradVar"]

    with open(folderPrefix+'Reinforce/'+envName+'ReinforceRewards_'+config_name+'_nRuns'+str(nRuns)+'_nEpochs'+str(nEpochs)+'_nTraj'+str(nTraj)+'_'+nameAppendix+'.pkl','wb') as f:
        pickle.dump({'evalPerEpochs': config.eval_per_epochs,\
                    'meanRewards': logsReinf,\
                    'evalRewards': evalReinf,\
                    'times': timesReinf,\
                    'trajLens': trajLensReinf,\
                    'gradVar': gradVarReinf,\
                    'gradVarMeanRewards': gradVarRewards},f)
