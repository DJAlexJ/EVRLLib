import numpy as np
import torch
import tqdm
import gc
import itertools
from ..simulators import pythonSimulators as pySim


class Error(Exception):
    pass


class AgentConfigError(Error):
    """
    Exception raised when agent config is invalid
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Reinforce:
    # implements Reinforce Agent (classic)
    def __init__(self, list_policy_net, simulator, n_trajectories=1, policy="gaussian",
                 device="cpu", nTrajectoriesForGradVar=1):
        """
        Args:
            List[torch.nn.Module] list_policy_net -- neural nets modelling the policy
            Simulator simulator -- simulator object
            int n_trajectories -- number of trajectories in MC estimate of the gradient
            str policy -- distribution to sample actions for continuous env
            str device -- device to use with torch
            int nTrajectoriesForGradVar -- number of trajectories to evaluate gradient variance
        """

        if n_trajectories < 1 or np.abs(n_trajectories-np.round(n_trajectories)) > 1e-14:
            raise AgentConfigError("", "The number of trajectories must be integer and strictly positive")
        if policy != "gaussian" and policy != "kumaraswamy":
            raise AgentConfigError("", "Policy distribution has incorrect value")

        self.policyNets = list_policy_net 
        self.simulator = simulator
        self.device = device
        self.policyDistr = policy
        for model in self.policyNets:
            model.to(self.device)
        self.nTrajectories = n_trajectories
        self.nTrajectoriesForGradVar = nTrajectoriesForGradVar
        
    def SampleActionDiscrete(self, state):
        """
        Samples actions based on the given state.
        Exploits softmax policy modelled by net[0]
        Args:
            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        try:
            policy = self.policyNets[0](state.to(self.device))  # if state is torch tensor
        except:
            # if state is numpy ndarray
            policy = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
            
        policyNumpy = policy.cpu().detach().numpy()
        if len(policy.shape) > 1:
            actions = [np.random.choice(np.arange(self.simulator.actionSpaceN),
                                        p=pol) for pol in policyNumpy]
            return np.array(actions), torch.log(policy)
        else:
            action = np.random.choice(np.arange(self.simulator.actionSpaceN),
                                      p=policyNumpy)
            return action, torch.log(torch.unsqueeze(policy[action], dim=0))
        
    def SampleActionContinuous(self, state):
        """
        Samples actions based on the given state
        if gaussian, sampler exploits gaussian policy with mean given by net[0]
            and diagonal covariance set by net[1]
        if kumaraswami, sampler exploits following strategy:
            if F - cdf of distribution x and u - uniform random variable, then
            F^-1(u) - random variable of x.
            a and b - kumaraswami parameters

        Args:
            state float32 [batch,stateDim] OR [stateDim] -- given state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        if self.policyDistr == "gaussian":
            try:
                mean = self.policyNets[0](state.to(self.device))
                std = self.policyNets[1](state.to(self.device))
            except:
                mean = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                std = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            std = torch.exp(std) + 1e-7
            if len(mean.shape) > 1:
                action = torch.normal(0, 1, size=mean.shape) * std + mean
                logprob = -0.5*torch.sum(torch.log(std), 1) - \
                          torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                          torch.sum(((action.detach() - mean)**2 * (1/std)), 1) / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
            else:
                action = torch.normal(0, 1, size=(self.simulator.actionSpaceShape[0],)) * std + mean
                logprob = -0.5*torch.sum(torch.log(std)) - torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                    ((action.detach() - mean)**2 * (1/std)).sum() / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
            
        elif self.policyDistr == "kumaraswamy":
            try:
                a = self.policyNets[0](state.to(self.device))
                b = self.policyNets[1](state.to(self.device))
            except:
                a = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                b = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            a = a + 0.1
            b = b + 0.1
            u = torch.from_numpy(np.random.uniform(0, 1, size=a.shape))
            action = (1-(1-u)**(1/b))**(1/a)
            logprob = torch.prod(torch.log(a) + torch.log(b) + (a-1)*torch.log(torch.abs(action.detach() + 1e-2)) + \
                                 (b-1)*torch.log(1 - torch.abs(action.detach() - 1e-2)**a)).view(1)

            high = self.simulator.gymInstance.action_space.high
            low = self.simulator.gymInstance.action_space.low

            act = action.clone().cpu().detach().numpy()
            act = (act - low)/(high - low)
            return act, logprob
   
    
    def train(self, n_epochs=2000, max_step=1000, lr=1e-3, eval_func=None, 
              eval_per_epochs=50, step_size=50, gamma=0.95, eps=0., entropy_const=0.0,
              verbose=0, count_grad_variance=-1):
        """ Trains the agent
        Args:
            int n_epochs -- number of epochs
            int max_step -- maximum length of sampled trajectory from the simulator
            float lr -- learning rate, parameter of the optimizer
            function eval_func -- function for evaluation
                accepts current agent and return statistics after agent.evaluate(n_samples, max_step)
            int eval_per_epochs -- perform eval_func each eval_per_epochs epochs
            int step_size -- scheduler stepsize, parameter of the optimizer
            float gamma -- discounting factor
            float eps -- epsilon-greedy parameter, 0 means greedy
            float entropy_const -- const by which the policy entropy is multiplied
            int verbose -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
            int count_grad_variance -- count gradient variance each count_grad_variance epochs

        Returns:
            dict stats -- {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}
        """
        
        stats = {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}

        # definition of the optimizers
        policyOptimizers = [torch.optim.Adam(Net.parameters(), lr=lr) for Net in self.policyNets]
        policySchedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                            for optimizer in policyOptimizers]
        
        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
            def policyHandler(state):
                action, logProb = self.SampleActionDiscrete(state)
                return action, logProb
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, logProb = self.SampleActionContinuous(state)
                return action, logProb

        # reset sampler for the simulator
        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)

        print("Training Reinforce....")
        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            # simulation
            states, actions, rewards, trajLens, logProbs = \
                self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                  self.nTrajectories, returnLogProbs=True,
                                                                  maxIterations=max_step
                                                                  )
            
            rewards = rewards[:, 0, :]  # assume 1-dimensional rewards

            # zeroPadding of logProbs
            maxLen = np.amax(trajLens)-1
            logProbs = [logProbs[k]+[torch.zeros(1)]*((maxLen-len(logProbs[k])).astype('int64')) if len(logProbs[k]) < maxLen \
                                                                               else logProbs[k] for k in np.arange(len(logProbs))]
            logProbsTensor = torch.cat([torch.unsqueeze(torch.cat(logProbs[k], 0), dim=0) for k in np.arange(len(logProbs))], 0)
            
            gammas = gamma**np.arange(rewards.shape[1])
            discountedSumsRewards = np.cumsum((gammas[None, :] * rewards)[:, ::-1], axis=1)[:, ::-1]/gammas[None, :]
            discountedSumsRewards = torch.from_numpy(discountedSumsRewards.astype("float32"))   
            
            policyGoal = -torch.mean(torch.sum(torch.from_numpy(gammas[None, :]).to(self.device)*discountedSumsRewards*logProbsTensor, 1))  # Expected Reward Goal
            
            if entropy_const > 0:
                policy_entropy = -entropy_const*torch.mean(torch.sum(torch.exp(logProbsTensor)*logProbsTensor, 1))
                policyGoal = policyGoal - policy_entropy

            if count_grad_variance != -1 and epochId % count_grad_variance == 0:
                 # simulation
                statesEST, actionsEST, rewardsEST, trajLensEST, logProbsEST = \
                    self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                      self.nTrajectoriesForGradVar,
                                                                      returnLogProbs=True, maxIterations=max_step
                                                                      )

                rewardsEST = rewardsEST[:, 0, :]  # assume 1-dimensional rewards
                stats['meanRewardsGradVar'].append(np.mean(np.sum(rewardsEST, axis=1)))

                # zeroPadding of logProbs
                maxLen = np.amax(trajLensEST)-1
                logProbsEST = [logProbsEST[k]+[torch.zeros(1)]*((maxLen-len(logProbsEST[k])).astype("int64")) if len(logProbsEST[k]) < maxLen \
                                                                                   else logProbsEST[k] for k in np.arange(len(logProbsEST))]
                logProbsTensorEST = torch.cat([torch.unsqueeze(torch.cat(logProbsEST[k], 0), dim=0) for k in np.arange(len(logProbsEST))], 0)
                
                gammas = gamma**np.arange(rewardsEST.shape[1])
                discountedSumsRewardsEST = np.cumsum((gammas[None, :] * rewardsEST)[:, ::-1], axis=1)[:, ::-1]/gammas[None, :]
                discountedSumsRewardsEST = torch.from_numpy(discountedSumsRewardsEST.astype("float32"))

                policyGoalsOLD = torch.sum(torch.from_numpy(gammas[None, :]).to(self.device) * discountedSumsRewardsEST * logProbsTensorEST, 1)

                autogradGrads=[[torch.autograd.grad(policyGoalsOLD[k], self.policyNets[pId].parameters(), retain_graph=True, create_graph=False)\
                                for pId in np.arange(len(self.policyNets))] for k in np.arange(self.nTrajectoriesForGradVar)]
                autogradGrads = list(itertools.chain.from_iterable(autogradGrads))

                rewardGradients = torch.cat([ \
                                    torch.unsqueeze(torch.cat([ 
                                                torch.flatten(grad) for grad in\
                                                    autogradGrads[k] \
                                    ], 0), 0)\
                                  for k in np.arange(self.nTrajectoriesForGradVar)], 0)

                valueGoalVar = torch.mean(torch.sum(rewardGradients*rewardGradients, 1))  # mean square norm, term1
                valueGoalVar = valueGoalVar - torch.sum(torch.mean(rewardGradients, 0)**2)
                valueGoalVar = valueGoalVar.detach()
                
                del rewardGradients, autogradGrads, policyGoalsOLD, discountedSumsRewardsEST, logProbsTensorEST, logProbsEST
                gc.collect()
            
                stats['GradientVariance'].append(valueGoalVar.item())

            # grad steps
            policyGoal.to(self.device).backward(retain_graph=True)
            for optimizer, scheduler in zip(policyOptimizers, policySchedulers):              
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            stats['meanRewards'][epochId] = np.mean(np.sum(rewards, axis=1))
            stats['nSteps'][epochId] = np.mean(trajLens-1)
            stats['policyGoal'][epochId] = policyGoal.item()

            if verbose > 0:
                if epochId % verbose == 0:
                    print("......", "meanReward:", stats['meanRewards'][epochId])
                    
            if eval_func is not None and epochId % eval_per_epochs == 0:
                eval_results = eval_func(self)
                stats['evalInfo'].append(eval_results)
            stats['evalFreq'] = eval_per_epochs

        print("DONE")
        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """ Evaluates the agent by sampling from the environment
        Args:
            int n_samples -- number of trajectories for estimation
            int max_step -- maximum length of trajectory allowed in simulator

        Returns:
            dict stats -- {'rewardMean': float, 'rewardStd': float}
        """
        stats = {'rewardMean': 0, 'rewardStd': 0}

        # policy handler for the simulator
        if self.simulator.actionSpaceType == 'Discrete':
             def policyHandler(state):      
                action, _ = self.SampleActionDiscrete(state)
                return action
            
        elif self.simulator.actionSpaceType == 'Continuous':
            def policyHandler(state):            
                action, _ = self.SampleActionContinuous(state)
                return action

        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)
        rewards = self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                    n_samples, returnRewards=True,
                                                                    maxIterations=max_step
                                                                    )
        stats['rewardMean'] = np.mean(np.sum(rewards[:, 0, :], axis=1))
        stats['rewardStd'] = np.std(np.sum(rewards[:, 0, :], axis=1))
            
        return stats
    

##################################################################

class ReinforceA2CBaseline:
    # implements Reinforce Agent with A2C baseline learned with least squares

    def __init__(self, list_policy_net, value_net, simulator, n_trajectories=1, policy='gaussian', device='cpu',
                nTrajectoriesForGradVar=1):
        """
        Args:
            List[torch.nn.Module] list_policy_net -- neural nets modelling the policy
            torch.nn.Module value_net -- neural net for baseline
            Simulator simulator -- simulator object
            int n_trajectories -- number of trajectories in MC estimate of the gradient
            str policy -- distribution to sample actions for continuous env
            str device -- device to use with torch
            int nTrajectoriesForGradVar -- number of trajectories to evaluate gradient variance
        """

        if n_trajectories < 1 or np.abs(n_trajectories-np.round(n_trajectories)) > 1e-14:
            raise AgentConfigError("", "The number of trajectories must be integer and strictly positive")
            
        if policy != "gaussian" and policy != "kumaraswamy":
            raise AgentConfigError("", "Policy distribution has incorrect value")

        self.simulator = simulator
        self.device = device
        self.policyNets = list_policy_net 
        self.policyDistr = policy
        for model in self.policyNets:
            model.to(self.device) 
        self.valueNet = value_net
        self.valueNet.to(self.device)
        self.nTrajectories = n_trajectories
        self.nTrajectoriesForGradVar = nTrajectoriesForGradVar
        
    def SampleActionDiscrete(self, state):
        """
        Samples actions based on the given state.
        Exploits softmax policy modelled by net[0]
        Args:
            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        try:
            policy = self.policyNets[0](state.to(self.device))  # if state is torch tensor
        except:
            # if state is numpy ndarray
            policy = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
            
        policyNumpy = policy.cpu().detach().numpy()
        if len(policy.shape) > 1:
            actions = [np.random.choice(np.arange(self.simulator.actionSpaceN),
                                        p=pol) for pol in policyNumpy]
            return np.array(actions), torch.log(policy)
        else:
            action = np.random.choice(np.arange(self.simulator.actionSpaceN),
                                      p=policyNumpy)
            return action, torch.log(torch.unsqueeze(policy[action], dim=0))
        
    def SampleActionContinuous(self, state):
        """
        Samples actions based on the given state
        if gaussian, sampler exploits gaussian policy with mean given by net[0]
            and diagonal covariance set by net[1]
        if kumaraswami, sampler exploits following strategy:
            if F - cdf of distribution x and u - uniform random variable, then
            F^-1(u) - random variable of x.
            a and b - kumaraswami parameters

        Args:
            state float32 [batch,stateDim] OR [stateDim] -- given state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        if self.policyDistr == "gaussian":
            try:
                mean = self.policyNets[0](state.to(self.device))
                std = self.policyNets[1](state.to(self.device))
            except:
                mean = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                std = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            std = torch.exp(std) + 1e-7
            if len(mean.shape) > 1:
                action = torch.normal(0, 1, size=mean.shape) * std + mean
                logprob = -0.5*torch.sum(torch.log(std), 1) - torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                    torch.sum(((action.detach() - mean)**2 * (1/std)), 1) / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
            else:
                action = torch.normal(0, 1, size=(self.simulator.actionSpaceShape[0],)) * std + mean
                logprob = -0.5*torch.sum(torch.log(std)) - torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                    ((action.detach() - mean)**2 * (1/std)).sum() / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
         
        elif self.policyDistr == "kumaraswamy":
            try:
                a = self.policyNets[0](state.to(self.device))
                b = self.policyNets[1](state.to(self.device))
            except:
                a = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                b = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            a = a + 0.1
            b = b + 0.1
            u = torch.from_numpy(np.random.uniform(0, 1, size=a.shape))
            action = (1-(1-u)**(1/b))**(1/a)
            logprob = torch.prod(torch.log(a) + torch.log(b) + (a-1)*torch.log(torch.abs(action.detach() + 1e-2)) +
                                 (b-1)*torch.log(1 - torch.abs(action.detach() - 1e-2)**a)).view(1)
            
            high = self.simulator.gymInstance.action_space.high
            low = self.simulator.gymInstance.action_space.low

            act = action.clone().cpu().detach().numpy()
            act = (act - low)/(high - low)
            return act, logprob

    def train(self, n_epochs=2000, max_step=1000, lr=1e-3, eval_func=None, 
              eval_per_epochs=50, step_size=50, gamma=0.95, eps=0., 
              entropy_const=0.0, verbose=0, count_grad_variance=-1):   
        """ Trains the agent
        Args:
            int n_epochs -- number of epochs
            int max_step -- maximum length of sampled trajectory from the simulator
            float lr -- learning rate, parameter of the optimizer
            function eval_func -- function for evaluation
                accepts current agent and return statistics after agent.evaluate(n_samples, max_step)
            int eval_per_epochs -- perform eval_func each eval_per_epochs epochs
            int step_size -- scheduler stepsize, parameter of the optimizer
            float gamma -- discounting factor
            float eps -- epsilon-greedy parameter, 0 means greedy
            float entropy_const -- const by which the policy entropy is multiplied
            int verbose -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
            int count_grad_variance -- count gradient variance each count_grad_variance epochs

        Returns:
            dict stats -- {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}
        """

        stats = {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}


        # Definition of the optimizers
        policyOptimizers = [torch.optim.Adam(Net.parameters(), lr=lr) for Net in self.policyNets]
        policySchedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                            for optimizer in policyOptimizers]

        valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=lr)
        valueScheduler = torch.optim.lr_scheduler.StepLR(valueOptimizer, step_size=step_size, gamma=gamma)


        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
             def policyHandler(state):      
                action, logProb = self.SampleActionDiscrete(state)
                return action, logProb
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, logProb = self.SampleActionContinuous(state)
                return action, logProb

        # reset sampler for the simulator
        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)

        print("Training Reinforce with A2CBaseline.....")
        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            # simulation
            states, actions, rewards, trajLens, logProbs = \
                self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                  self.nTrajectories, returnLogProbs=True,
                                                                  maxIterations=max_step
                                                                  )
            rewards = rewards[:, 0, :]  # assume 1-dimensional rewards

            # zeroPadding of logProbs
            maxLen = np.amax(trajLens)-1
            logProbs = [logProbs[k]+[torch.zeros(1)]*((maxLen-len(logProbs[k])).astype('int64')) if len(logProbs[k]) < maxLen \
                                                                               else logProbs[k]
                        for k in np.arange(len(logProbs))]
            logProbsTensor = torch.cat([torch.unsqueeze(torch.cat(logProbs[k], 0), dim=0)
                                        for k in np.arange(len(logProbs))], 0)
            
            gammas = gamma**np.arange(rewards.shape[1])
            discountedSumsRewardsInit = np.cumsum((gammas[None, :] * rewards)[:, ::-1], axis=1)[:, ::-1]/gammas[None, :]

            baselines = torch.squeeze(self.valueNet(torch.from_numpy(np.transpose(states[:, :, :-1].astype("float32"), (0, 2, 1)))), dim=-1)
            
            discountedSumsRewards = torch.from_numpy(discountedSumsRewardsInit.astype("float32")) - baselines.detach()
            
            policyGoal = -torch.mean(torch.sum(torch.from_numpy(gammas[None, :])*discountedSumsRewards * logProbsTensor, 1))  # Expected Reward Goal
            if entropy_const > 0:
                policy_entropy = -entropy_const*torch.mean(torch.sum(torch.exp(logProbsTensor)*logProbsTensor, 1))
                policyGoal = policyGoal - policy_entropy

            if count_grad_variance != -1 and epochId % count_grad_variance == 0:
                # simulation
                statesEST, actionsEST, rewardsEST, trajLensEST, logProbsEST = \
                    self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                      self.nTrajectoriesForGradVar,
                                                                      returnLogProbs=True, maxIterations=max_step
                                                                      )

                rewardsEST = rewardsEST[:, 0, :]  # assume 1-dimensional rewards
                stats['meanRewardsGradVar'].append(np.mean(np.sum(rewardsEST, axis=1)))
                
                # zeroPadding of logProbs
                maxLen = np.amax(trajLensEST)-1

                logProbsEST = [logProbsEST[k]+[torch.zeros(1)]*((maxLen-len(logProbsEST[k])).astype('int64'))
                               if len(logProbsEST[k]) < maxLen else logProbsEST[k]
                               for k in np.arange(len(logProbsEST))]

                logProbsTensorEST = torch.cat([torch.unsqueeze(torch.cat(logProbsEST[k], 0), dim=0)
                                               for k in np.arange(len(logProbsEST))], 0)

                gammas = gamma**np.arange(rewardsEST.shape[1])
                discountedSumsRewardsEST = np.cumsum((gammas[None, :]*rewardsEST)[:, ::-1], axis=1)[:, ::-1]/gammas[None, :]
        
                baselinesEST = torch.squeeze(self.valueNet(torch.from_numpy(np.transpose(statesEST[:, :, :-1].astype("float32"), (0, 2, 1)))), dim=-1)
                discountedSumsRewardsEST = torch.from_numpy(discountedSumsRewardsEST.astype("float32")) - baselinesEST
                policyGoalsEST = torch.sum(torch.from_numpy(gammas[None, :])*discountedSumsRewardsEST * logProbsTensorEST, 1)

                autogradGrads=[[torch.autograd.grad(
                    policyGoalsEST[k],
                    self.policyNets[pId].parameters(),
                    retain_graph=True,
                    create_graph=False)
                    for pId in np.arange(len(self.policyNets))]
                    for k in np.arange(self.nTrajectoriesForGradVar)]
                autogradGrads = list(itertools.chain.from_iterable(autogradGrads))

                rewardGradients = torch.cat([ \
                                    torch.unsqueeze(torch.cat([ 
                                                torch.flatten(grad) for grad in\
                                                    autogradGrads[k]\
                                    ], 0), 0)\
                                  for k in np.arange(self.nTrajectoriesForGradVar)], 0)

                valueGoalVar = torch.mean(torch.sum(rewardGradients*rewardGradients, 1))  # mean square norm, term1
                valueGoalVar = valueGoalVar - torch.sum(torch.mean(rewardGradients, 0)**2)
                valueGoalVar = valueGoalVar.detach()
                
                del rewardGradients, autogradGrads, policyGoalsEST, discountedSumsRewardsEST
                del baselinesEST, logProbsTensorEST, logProbsEST
                gc.collect()
            
                stats['GradientVariance'].append(valueGoalVar.item())
                
            valueGoal = -torch.tensor([2])*torch.mean(torch.sum(baselines*discountedSumsRewards, 1))  # Least squares goal

            # grad steps
            policyGoal.to(self.device).backward(retain_graph=True)
            for optimizer, scheduler in zip(policyOptimizers, policySchedulers):              
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            valueGoal.to(self.device).backward()
            valueOptimizer.step()
            valueScheduler.step()
            valueOptimizer.zero_grad()
            
            stats['meanRewards'][epochId] = np.mean(np.sum(rewards, axis=1))
            stats['nSteps'][epochId] = np.mean(trajLens-1)
            stats['policyGoal'][epochId] = policyGoal.item()
            stats['valueGoal'][epochId] = valueGoal.item()

            if verbose > 0:
                if epochId % verbose == 0:
                    print("......", "meanReward:", stats['meanRewards'][epochId])
                    
            if eval_func is not None and epochId % eval_per_epochs == 0:
                eval_results = eval_func(self)
                stats['evalInfo'].append(eval_results)
            stats['evalFreq'] = eval_per_epochs

        print("DONE")
        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """ Evaluates the agent by sampling from the environment
        Args:
            int n_samples -- number of trajectories for estimation
            int max_step -- maximum length of trajectory allowed in simulator

        Returns:
            dict stats -- {'rewardMean': float, 'rewardStd': float}
        """
        
        stats = {'rewardMean': 0, 'rewardStd': 0}

        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
             def policyHandler(state):      
                action, _ = self.SampleActionDiscrete(state)
                return action
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, _ = self.SampleActionContinuous(state)
                return action

        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)
        rewards = self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                    n_samples, returnRewards=True,
                                                                    maxIterations=max_step
                                                                    )
        stats["rewardMean"] = np.mean(np.sum(rewards[:, 0, :], axis=1))
        stats["rewardStd"] = np.std(np.sum(rewards[:, 0, :], axis=1))
            
        return stats


##################################################################


class ReinforceEVBaseline:
    # implements Reinforce Agent with A2C baseline learned with least squares

    def __init__(self, list_policy_net, value_net, simulator, n_trajectories=1, policy='gaussian', device='cpu', baseline_loss='2ndMoment', nTrajectoriesForGradVar=1):
        """
        Args:
            List[torch.nn.Module] list_policy_net -- neural nets modelling the policy
            torch.nn.Module value_net -- neural net for baseline
            Simulator simulator -- simulator object
            int n_trajectories -- number of trajectories in MC estimate of the gradient
            str policy -- distribution to sample actions for continuous env
            str device -- device to use with torch
            str baseline_loss -- loss to use for baseline training
                var: full empirical variance
                2ndMoment: only the second moment
            int nTrajectoriesForGradVar -- number of trajectories to evaluate gradient variance
        """
        if (n_trajectories < 1 or np.abs(n_trajectories-np.round(n_trajectories)) > 1e-14):
            raise AgentConfigError("", "The number of trajectories must be integer and strictly positive")
           
        if (policy != "gaussian" and policy != "kumaraswamy"):
            raise AgentConfigError("", "Policy distribution has incorrect value")

        if (not (baseline_loss == "var" or baseline_loss == "2ndMoment")):
            raise AgentConfigError("if(not (baseline_loss=='var' or baseline_loss=='2ndMoment') ):",
                                   " Baseline loss should be either 'var' or '2ndMoment'")

        if (baseline_loss == "var" and n_trajectories == 1):
            raise AgentConfigError("if(baseline_loss=='var' and n_trajectories==1):",
                                   " If Baseline loss is variance, the number of trajectories should be more than 1'")

        self.simulator = simulator
        self.device = device
        self.policyNets = list_policy_net 
        self.policyDistr = policy
        for model in self.policyNets:
            model.to(self.device) 
        self.valueNet = value_net
        self.valueNet.to(self.device)
        self.nTrajectories = n_trajectories
        self.nTrajectoriesForGradVar = nTrajectoriesForGradVar
        self.baselineLoss = baseline_loss


    def SampleActionDiscrete(self, state):
        """
        Samples actions based on the given state.
        Exploits softmax policy modelled by net[0]
        Args:
            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        try:
            policy = self.policyNets[0](state.to(self.device))  # if state is torch tensor
        except:
            policy = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))  # if state is numpy ndarray
            
        policyNumpy = policy.cpu().detach().numpy()
        if len(policy.shape) > 1:
            actions = [np.random.choice(np.arange(self.simulator.actionSpaceN),
                                        p=pol) for pol in policyNumpy]
            return np.array(actions), torch.log(policy)
        else:
            action=np.random.choice(np.arange(self.simulator.actionSpaceN),
                                    p=policyNumpy)
            return action, torch.log(torch.unsqueeze(policy[action], dim=0))
        

    def SampleActionContinuous(self, state):
        """
        Samples actions based on the given state
        if gaussian, sampler exploits gaussian policy with mean given by net[0]
            and diagonal covariance set by net[1]
        if kumaraswami, sampler exploits following strategy:
            if F - cdf of distribution x and u - uniform random variable, then
            F^-1(u) - random variable of x.
            a and b - kumaraswami parameters

        Args:
            state float32 [batch,stateDim] OR [stateDim] -- given state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        if self.policyDistr == "gaussian":
            try:
                mean = self.policyNets[0](state.to(self.device))
                std = self.policyNets[1](state.to(self.device))
            except:
                mean = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                std = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            std = torch.exp(std) + 1e-7
            if len(mean.shape) > 1:
                action = torch.normal(0, 1, size=mean.shape) * std + mean
                logprob = -0.5*torch.sum(torch.log(std), 1) - torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                    torch.sum(((action.detach() - mean)**2 * (1/std)), 1) / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
            else:
                action = torch.normal(0, 1, size=(self.simulator.actionSpaceShape[0],)) * std + mean
                logprob = -0.5*torch.sum(torch.log(std)) - torch.tensor([len(std)/2])*torch.log(torch.as_tensor([2 * np.pi])) - \
                    ((action.detach() - mean)**2 * (1/std)).sum() / torch.tensor([2])

                a = action.clone().cpu().detach().numpy()
                return a, logprob
         
        elif self.policyDistr == "kumaraswamy":
            try:
                a = self.policyNets[0](state.to(self.device))
                b = self.policyNets[1](state.to(self.device))
            except:
                a = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
                b = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))

            a = a + 0.1
            b = b + 0.1
            u = torch.from_numpy(np.random.uniform(0, 1, size=a.shape))
            action = (1-(1-u)**(1/b))**(1/a)
            logprob = torch.prod(torch.log(a) + torch.log(b) + (a-1)*torch.log(torch.abs(action.detach() + 1e-2)) + \
                                 (b-1)*torch.log(1 - torch.abs(action.detach() - 1e-2)**a)).view(1)
            
            high = self.simulator.gymInstance.action_space.high
            low = self.simulator.gymInstance.action_space.low

            act = action.clone().cpu().detach().numpy()
            act = (act - low)/(high - low)
            return act, logprob
    
    def train(self, n_epochs=2000, max_step=1000, lr=1e-3, eval_func=None, eval_per_epochs=50, step_size=50, gamma=0.95, eps=0., entropy_const=0.0, verbose=0, count_grad_variance=-1):   
        """ Trains the agent
        Args:
            int n_epochs -- number of epochs
            int max_step -- maximum length of sampled trajectory from the simulator
            float lr -- learning rate, parameter of the optimizer
            function eval_func -- function for evaluation
                accepts current agent and return statistics after agent.evaluate(n_samples, max_step)
            int eval_per_epochs -- perform eval_func each eval_per_epochs epochs
            int step_size -- scheduler stepsize, parameter of the optimizer
            float gamma -- discounting factor
            float eps -- epsilon-greedy parameter, 0 means greedy
            float entropy_const -- const by which the policy entropy is multiplied
            int verbose -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
            int count_grad_variance -- count gradient variance each count_grad_variance epochs

        Returns:
            dict stats -- {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}
        """

        stats = {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs]),
                 "evalInfo": [], "evalFreq": 0, "GradientVariance": [], "meanRewardsGradVar": []}


        #Definition of the optimizers
        policyOptimizers = [torch.optim.Adam(Net.parameters(), lr=lr) for Net in self.policyNets]
        policySchedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) for optimizer in policyOptimizers]

        valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=lr)
        valueScheduler = torch.optim.lr_scheduler.StepLR(valueOptimizer, step_size=step_size, gamma=gamma)


        #policy handler for the simulator
        if self.simulator.actionSpaceType == 'Discrete':
             def policyHandler(state):      
                action, logProb = self.SampleActionDiscrete(state)
                return action, logProb
            
        elif self.simulator.actionSpaceType == 'Continuous':
            def policyHandler(state):            
                action, logProb = self.SampleActionContinuous(state)
                return action, logProb

        #reset sampler for the simulator
        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)


        print("Training Reinforce with EVBaseline(loss="+self.baselineLoss+").....")
        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            #simulation
            states,actions,rewards,trajLens, logProbs = \
                self.simulator.SampleTrajectoriesFromStateSampler( stateSampler, policyHandler,self.nTrajectories,\
                      returnLogProbs=True, maxIterations=max_step)
            rewards = rewards[:,0,:] #!!assume 1-dimensional rewards

            #zeroPadding of logProbs
            maxLen=np.amax(trajLens)-1
            logProbs = [logProbs[k]+[torch.zeros(1)]*((maxLen-len(logProbs[k])).astype('int64')) if len(logProbs[k])<maxLen \
                                                                               else logProbs[k] for k in np.arange(len(logProbs))]
            logProbsTensor = torch.cat([torch.unsqueeze(torch.cat(logProbs[k],0),dim=0) for k in np.arange(len(logProbs))],0)
            
            gammas = gamma**np.arange(rewards.shape[1])
            discountedSumsRewardsNP=np.cumsum((gammas[None,:] *rewards)[:,::-1], axis=1)[:,::-1]/gammas[None,:]


            #VALUE PREPARATION
            #OLD means what is currently there, when policy and value are not yet updated
            baselinesOLD = torch.squeeze(self.valueNet(torch.from_numpy(np.transpose(states[:,:,:-1].astype("float32"),(0,2,1) ))), dim=-1)
            baselinesOLDDETACHED = baselinesOLD.detach()

            discountedSumsRewardsOLD = torch.from_numpy(discountedSumsRewardsNP.astype("float32")) - baselinesOLD
            
            policyGoalsOLD = torch.sum(torch.from_numpy(gammas[None,:])*discountedSumsRewardsOLD * logProbsTensor,1)

            autogradGrads=[ [torch.autograd.grad(policyGoalsOLD[k],self.policyNets[pId].parameters(), retain_graph=True, create_graph=True)\
                            for pId in np.arange(len(self.policyNets))] for k in np.arange(self.nTrajectories)]
            autogradGrads = list(itertools.chain.from_iterable(autogradGrads))

            rewardGradients = torch.cat([ \
                                torch.unsqueeze(torch.cat([ 
                                            torch.flatten(grad) for grad in\
                                                autogradGrads[k]\
                                ],0),0)\
                              for k in np.arange(self.nTrajectories)],0)

            valueGoal = torch.mean(torch.sum(rewardGradients*rewardGradients,1)) #mean square norm, term1
            if (self.baselineLoss=="var"):
                #if it is, add one more term
                valueGoal = valueGoal - torch.sum(torch.mean(rewardGradients,0)**2)
            
            if count_grad_variance != -1 and epochId % count_grad_variance == 0:
                # simulation
                statesEST, actionsEST, rewardsEST, trajLensEST, logProbsEST = \
                    self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                  self.nTrajectoriesForGradVar, returnLogProbs=True,
                                                                  maxIterations=max_step)

                rewardsEST = rewardsEST[:, 0, :]  # assume 1-dimensional rewards
                stats['meanRewardsGradVar'].append(np.mean(np.sum(rewardsEST, axis=1)))
                
                # zeroPadding of logProbs
                maxLen = np.amax(trajLensEST)-1

                logProbsEST = [logProbsEST[k]+[torch.zeros(1)]*((maxLen-len(logProbsEST[k])).astype('int64'))
                               if len(logProbsEST[k]) < maxLen else logProbsEST[k]
                               for k in np.arange(len(logProbsEST))]

                logProbsTensorEST = torch.cat([torch.unsqueeze(torch.cat(logProbsEST[k], 0), dim=0)
                                               for k in np.arange(len(logProbsEST))], 0)

                gammasEST = gamma**np.arange(rewardsEST.shape[1])
                discountedSumsRewardsEST = np.cumsum((gammasEST[None, :]*rewardsEST)[:, ::-1], axis=1)[:, ::-1]/gammasEST[None, :]
                    
                baselinesEST = torch.squeeze(self.valueNet(torch.from_numpy(np.transpose(statesEST[:, :, :-1].astype("float32"), (0, 2, 1)))), dim=-1)
                discountedSumsRewardsEST = torch.from_numpy(discountedSumsRewardsEST.astype("float32")) - baselinesEST
                policyGoalsEST = torch.sum(torch.from_numpy(gammasEST[None, :])*discountedSumsRewardsEST * logProbsTensorEST, 1)
                
                autogradGradsEST = [[torch.autograd.grad(
                    policyGoalsEST[k],
                    self.policyNets[pId].parameters(),
                    retain_graph=True,
                    create_graph=False)
                    for pId in np.arange(len(self.policyNets))]
                    for k in np.arange(self.nTrajectoriesForGradVar)
                ]
                autogradGradsEST = list(itertools.chain.from_iterable(autogradGradsEST))

                rewardGradientsEST = torch.cat([ \
                                    torch.unsqueeze(torch.cat([ 
                                                torch.flatten(grad) for grad in\
                                                    autogradGradsEST[k]\
                                    ], 0), 0)\
                                  for k in np.arange(self.nTrajectoriesForGradVar)], 0)

                valueGoalVar = torch.mean(torch.sum(rewardGradientsEST*rewardGradientsEST, 1))  # mean square norm, term1
                valueGoalVar = valueGoalVar - torch.sum(torch.mean(rewardGradientsEST, 0)**2)
                valueGoalVar = valueGoalVar.detach()
                
                del rewardGradientsEST, autogradGradsEST, policyGoalsEST, discountedSumsRewardsEST
                del baselinesEST, logProbsTensorEST, logProbsEST, gammasEST
                gc.collect()
            
                stats["GradientVariance"].append(valueGoalVar.item())


            # VALUE STEP
            valueGoal.to(self.device).backward(retain_graph=True)
            valueOptimizer.step()
            valueScheduler.step()
            # clean it before policy update
            for optimizer in policyOptimizers:              
                optimizer.zero_grad()


            # POLICY STEP
            discountedSumsRewards = torch.from_numpy(discountedSumsRewardsNP.astype("float32")) - baselinesOLDDETACHED
            policyGoals = torch.sum(torch.from_numpy(gammas[None,:])*discountedSumsRewards * logProbsTensor, 1)
            policyGoal = -torch.mean(policyGoals)
            
            if entropy_const > 0:
                policy_entropy = -entropy_const*torch.mean(torch.sum(torch.exp(logProbsTensor)*logProbsTensor, 1))
                policyGoal = policyGoal - policy_entropy

            policyGoal.to(self.device).backward(retain_graph=True)
            for optimizer, scheduler in zip(policyOptimizers, policySchedulers):              
                optimizer.step()
                scheduler.step()

            # Overall clean-up
            valueOptimizer.zero_grad()
            for optimizer in policyOptimizers:              
                optimizer.zero_grad()

                   
            stats["meanRewards"][epochId] = np.mean(np.sum(rewards, axis=1))
            stats["nSteps"][epochId] = np.mean(trajLens-1)
            stats["policyGoal"][epochId] = policyGoal.item()
            stats["valueGoal"][epochId] = valueGoal.item()

            if verbose > 0:
                if epochId % verbose == 0:
                    print("......", "meanReward:", stats['meanRewards'][epochId])
                    
            if eval_func is not None and epochId % eval_per_epochs == 0:
                eval_results = eval_func(self)
                stats['evalInfo'].append(eval_results)
            stats['evalFreq'] = eval_per_epochs

        print("DONE")
        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """ Evaluates the agent by sampling from the environment
        Args:
            int n_samples -- number of trajectories for estimation
            int max_step -- maximum length of trajectory allowed in simulator

        Returns:
            dict stats -- {'rewardMean': float, 'rewardStd': float}
        """
        
        stats = {"rewardMean": 0, "rewardStd": 0}

        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
             def policyHandler(state):      
                action, _ = self.SampleActionDiscrete(state)
                return action
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, _ = self.SampleActionContinuous(state)
                return action

        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)
        rewards = self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                  n_samples, returnRewards=True,
                                                                  maxIterations=max_step
                                                                  )
        stats['rewardMean'] = np.mean(np.sum(rewards[:, 0, :], axis=1))
        stats['rewardStd'] = np.std(np.sum(rewards[:, 0, :], axis=1))
            
        return stats
        
##################################################################

class ReinforceWeightedA2CBaseline:
    # implements Reinforce Agent with A2C baseline learned with weighted least squares

    def __init__(self, list_policy_net, value_net, simulator, n_trajectories=1, device='cpu'):
        """
        Args:
            List[torch.nn.Module] list_policy_net -- neural nets modelling the policy
            torch.nn.Module value_net --  neural net for baseline
            Simulator simulator -- simulator object
            int n_trajectories -- number of trajectories in MC estimate of the gradient
            str device -- device to use with torch
        """

        if n_trajectories < 1 or np.abs(n_trajectories-np.round(n_trajectories)) > 1e-14:
            raise AgentConfigError("", "The number of trajectories must be integer and strictly positive")

        self.simulator = simulator
        self.device = device
        self.policyNets = list_policy_net 
        self.device = device
        for model in self.policyNets:
            model.to(self.device) 
        self.valueNet = value_net
        self.valueNet.to(self.device)
        self.nTrajectories = n_trajectories
        
    def SampleActionDiscrete(self, state):
        """
        Samples actions based on the given state.
        Exploits softmax policy modelled by net[0]
        Args:
            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        try:
            policy = self.policyNets[0](state.to(self.device))  # if state is torch tensor
        except:
            policy = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))  # if state is numpy ndarray
            
        policyNumpy = policy.cpu().detach().numpy()
        if len(policy.shape) > 1:
            actions = [np.random.choice(np.arange(self.simulator.actionSpaceN),
                                        p=pol) for pol in policyNumpy]
            return np.array(actions), torch.log(policy)
        else:
            action = np.random.choice(np.arange(self.simulator.actionSpaceN),
                                    p=policyNumpy)
            return action, torch.log(torch.unsqueeze(policy[action], dim=0))
        
    def SampleActionContinuous(self, state):
        """
        Samples actions based on the given state
        Sampler exploits gaussian policy with mean given by net[0]
        and diagonal covariance set by net[1]

        Args:
            state float32 [batch,stateDim] OR [stateDim] -- given state

        Returns:
            np.float32 a -- action to play
            torch.Tensor logprob -- logarithm of probabilities
        """
        try:
            mean = self.policyNets[0](state.to(self.device))
            std = self.policyNets[1](state.to(self.device))
        except:
            mean = self.policyNets[0](torch.from_numpy(state.astype("float32")).to(self.device))
            std = self.policyNets[1](torch.from_numpy(state.astype("float32")).to(self.device))
        
        if len(mean.shape) > 1:
            action = torch.normal(0, 1, size=mean.shape)*torch.exp(std) + mean
            logprob = -torch.log(torch.sum(torch.exp(std), 1) * torch.tensor([(2 * np.pi)**(std.shape[1]/2)])) - \
                      torch.sum((action.detach() - mean)**2 * (1/torch.exp(std)**2), 1) / torch.tensor([2])
            
            a = action.clone().cpu().detach().numpy()
            return a, logprob
        else:
            action = torch.normal(0, 1, size=(self.simulator.actionSpaceShape[0],)) * torch.exp(std) + mean
            logprob = -torch.log(torch.exp(std).sum() * torch.tensor([(2 * np.pi)**(std.shape[0]/2)])) - \
                      ((action.detach() - mean)**2 * (1/torch.exp(std)**2)).sum() / torch.tensor([2])
            
            a = action.clone().cpu().detach().numpy()
            return a, logprob
   
    def sqnormGrad(self, fx, x):
        # Computes scalar square of grad_x fx
        grads = [torch.autograd.grad(fx, x[k], retain_graph=True, only_inputs=True) for k in np.arange(len(x))]
        
        return np.sum(np.array([[np.sum(np.square(x.numpy().flatten())) for x in grad] for grad in grads]))


    def train(self, n_epochs=2000, max_step=1000, lr=1e-3, step_size=50, gamma=0.95, eps=0.,verbose=0):   
        """ Trains the agent
        Args:
            int n_epochs -- number of epochs
            int max_step -- maximum length of sampled trajectory from the simulator
            float lr -- learning rate, parameter of the optimizer
            int step_size -- scheduler stepsize, parameter of the optimizer
            float gamma -- discounting factor
            float eps -- epsilon-greedy parameter, 0 means greedy
            int verbose -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs

        Returns:
            dict stats -- {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs])
                 }
        """

        stats = {"meanRewards": np.zeros([n_epochs]), "nSteps": np.zeros([n_epochs]),
                 "policyGoal": np.zeros([n_epochs]), "valueGoal": np.zeros([n_epochs])
                 }


        # Definition of the optimizers
        policyOptimizers = [torch.optim.Adam(Net.parameters(), lr=lr) for Net in self.policyNets]
        policySchedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) for optimizer in policyOptimizers]

        valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=lr)
        valueScheduler = torch.optim.lr_scheduler.StepLR(valueOptimizer, step_size=step_size, gamma=gamma)

        # policy weights
        policyWeights=[self.policyNets[k].parameters() for k in np.arange(len(self.policyNets))]

        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
             def policyHandler(state):      
                action, logProb = self.SampleActionDiscrete(state)
                return action, logProb
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, logProb = self.SampleActionContinuous(state)
                return action, logProb

        # reset sampler for the simulator
        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)

        print("Training Reinforce with WA2CBaseline.....")
        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            # simulation
            states, actions, rewards, trajLens, logProbs = \
                self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                  self.nTrajectories, returnLogProbs=True,
                                                                  maxIterations=max_step
                                                                  )
            rewards = rewards[:, 0, :]  # assume 1-dimensional rewards
            trajLens = trajLens.astype("int64")

            # zeroPadding of logProbs
            maxLen = np.amax(trajLens) - 1
            logProbs = [logProbs[k]+[torch.zeros(1)]*((maxLen-len(logProbs[k])).astype("int64")) if len(logProbs[k]) < maxLen \
                                                                               else logProbs[k] for k in np.arange(len(logProbs))]
            logProbsTensor = torch.cat([torch.unsqueeze(torch.cat(logProbs[k], 0), dim=0) for k in np.arange(len(logProbs))], 0)
            
            gammas = gamma**np.arange(rewards.shape[1])
            discountedSumsRewards = np.cumsum((gammas[None, :]*rewards)[:, ::-1], axis=1)[:, ::-1]/gammas[None, :]

            baselines = torch.squeeze(self.valueNet(torch.from_numpy(np.transpose(states[:, :, :-1].astype("float32"), (0, 2, 1)))), dim=-1)
            discountedSumsRewards = torch.from_numpy(discountedSumsRewards.astype("float32")) - baselines.cpu().detach()
            # weights for baseline training
            
            baselineWeights = torch.tensor([[ self.sqnormGrad(logProbsTensor[kk, tt], policyWeights) for tt in np.arange(trajLens[kk]-1)] \
                                                    + [torch.zeros(1)]*((maxLen-trajLens[kk]+1).astype("int64"))
                                            for kk in np.arange(self.nTrajectories)])
            baselineWeights.requiresGrad = False

            policyGoal = -torch.mean(torch.sum(discountedSumsRewards * logProbsTensor, 1))  # Expected Reward Goal
            valueGoal = -torch.tensor([2])*torch.mean(torch.sum(torch.from_numpy(trajLens[:, None]-1)*baselineWeights*baselines*discountedSumsRewards, 1))  # weighted Least squares goal
            

            # grad steps
            valueGoal.to(self.device).backward()
            valueOptimizer.step()
            valueScheduler.step()
            valueOptimizer.zero_grad()

            policyGoal.to(self.device).backward(retain_graph=True)
            for optimizer, scheduler in zip(policyOptimizers, policySchedulers):              
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            
            stats["meanRewards"][epochId] = np.mean(np.sum(rewards, axis=1))
            stats["nSteps"][epochId] = np.mean(trajLens-1)
            stats["policyGoal"][epochId] = policyGoal.item()
            stats["valueGoal"][epochId] = valueGoal.item()

            if verbose > 0:
                if epochId % verbose == 0:
                    print("......", "meanReward:", stats['meanRewards'][epochId])

        print("DONE")
        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """ Evaluates the agent by sampling from the environment
        Args:
            int n_samples -- number of trajectories for estimation
            int max_step -- maximum length of trajectory allowed in simulator

        Returns:
            dict stats -- {'rewardMean': float, 'rewardStd': float}
        """
        
        stats = {"rewardMean": 0, "rewardStd": 0}

        # policy handler for the simulator
        if self.simulator.actionSpaceType == "Discrete":
             def policyHandler(state):      
                action, _ = self.SampleActionDiscrete(state)
                return action
            
        elif self.simulator.actionSpaceType == "Continuous":
            def policyHandler(state):            
                action, _ = self.SampleActionContinuous(state)
                return action

        stateSampler = pySim.GymResetSampler(self.simulator.gymInstance)
        rewards = self.simulator.SampleTrajectoriesFromStateSampler(stateSampler, policyHandler,
                                                                    n_samples, returnRewards=True,
                                                                    maxIterations=max_step
                                                                    )
        stats["rewardMean"] = np.mean(np.sum(rewards[:, 0, :], axis=1))
        stats["rewardStd"] = np.std(np.sum(rewards[:, 0, :], axis=1))
            
        return stats
