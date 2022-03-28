import os
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import rllib.rlagents.ReinforceA2C as srlagents
import argparse
import importlib
import time
import pickle

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

evaluation_A2C = config.evaluation_A2C
    
folderPrefix = config.folderPrefix
nameAppendix = 'lr'+str(config.lr)+'_'+str(np.random.randint(10050000))

logsA2C = np.zeros([nRuns, nEpochs])
evalA2C = np.zeros([nRuns, int(np.ceil(nEpochs/eval_per_epochs))])
timesA2C = np.zeros([nRuns])
trajLensA2C = np.zeros([nRuns, nEpochs])  # np.mean(trajLens) is returned so multiply by nTraj
gradVarA2C_w_baseline = np.zeros([nRuns, int(np.ceil(nEpochs/count_grad_var))])
gradVarA2C_wo_baseline = np.zeros([nRuns, int(np.ceil(nEpochs/count_grad_var))])
gradVarRewards = np.zeros([nRuns, int(np.ceil(nEpochs/count_grad_var))])

for runId in np.arange(nRuns):
    print("RUN ", runId)
    
    polval = config.PolicyValuePair()
    policyNets = polval.policyNet
    valueNet = polval.valueNet
    agent = srlagents.ReinforceA2CBaseline(policyNets, valueNet, simulator, 
                                           n_trajectories=nTraj, nTrajectoriesForGradVar=nTrajForGradVar)

    time0 = time.time()
    logs = agent.train(n_epochs=nEpochs, lr=lr, max_step=max_step, eval_func=evaluation_A2C,
                       step_size=step_size, gamma=gamma, entropy_const=entropy_const,
                       eval_per_epochs=eval_per_epochs, count_grad_variance=count_grad_var)
    timesA2C[runId] = time.time()-time0
    logsA2C[runId, :] = logs['meanRewards']
    evalA2C[runId, :] = list(map(lambda x: x['rewardMean'], logs['evalInfo']))
    trajLensA2C[runId, :] = logs['nSteps']*nTraj
    gradVarA2C_w_baseline[runId, :] = logs["GradientVariance_w_baseline"]
    gradVarA2C_wo_baseline[runId, :] = logs["GradientVariance_wo_baseline"]
    gradVarRewards[runId, :] = logs["meanRewardsGradVar"]

    with open(folderPrefix+'A2C/'+envName+'A2CRewards_'+config_name+'_nRuns'+str(nRuns)+'_nEpochs'+str(nEpochs)+'_nTraj'+str(nTraj)+'_'+nameAppendix+'.pkl', 'wb') as f:
        pickle.dump({
            'evalPerEpochs': config.eval_per_epochs,
            'meanRewards': logsA2C,
            'evalRewards': evalA2C,
            'times': timesA2C,
            'trajLens': trajLensA2C,
            'gradVar_w_baseline': gradVarA2C_w_baseline,
            'gradVar_wo_baseline': gradVarA2C_wo_baseline,
            'gradVarMeanRewards': gradVarRewards
        }, f)
