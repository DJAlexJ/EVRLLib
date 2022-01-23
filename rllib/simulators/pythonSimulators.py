import gym
import numpy as np

import multiprocessing as mp


class Error(Exception):
    pass



class ShapeMismatchError(Error):
    """Exception raised for shape mismatch in the input.
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class PolicyMismatchError(Error):
    """
        Exception raised for policy output mismatch.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class RegimeMismatchError(Error):
    """
        Exception raised for policy output mismatch.
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



class StateSampler:
    def __init__(self):
        pass

    def sample(self,Nsamples):
        #samples Nsamples
        #returns [NSamples,]
        pass



class GymResetSampler(StateSampler):

    def __init__(self, gymEnv):
        #gymEnv -- gym environment
        #be careful to create separate copy of it
        self.gymEnv = gymEnv
        self.stateSpaceType = 'Continuous'
        try:
            self.stateSpaceShape = gymEnv.reset().shape
            if(len(self.stateSpaceShape)==0):
                self.stateSpaceShape = (1,)
                self.stateSpaceN = self.gymEnv.observation_space.n
                self.stateSpaceType='Discrete'
        except:
            self.stateSpaceShape = (1,)
            self.stateSpaceN = self.gymEnv.observation_space.n
            self.stateSpaceType='Discrete'


    def sample(self,Nsamples=1):
        if(self.stateSpaceType=='Discrete'):
            return np.concatenate([np.array([self.gymEnv.reset()])[None,:] for k in np.arange(Nsamples)], axis=0)
        else:
            return np.concatenate([self.gymEnv.reset()[None,:] for k in np.arange(Nsamples)], axis=0)



class GymSimulator:
    """
    Basic gym Simulator class with sampling, visualization and info routines.

    !!supports only Box and Discrete action and state spaces! If you want Tensor state, there is a separate GymTensorStateSimulator.
    """
    

    def __init__(self,gymInst):
        """
        Initializes GymSimulator
        gymInst -- gym Instance (environment)
        """
        #gymInstance -- gym environment
        #the rest are its parameters
        self.gymInstance = gymInst

        self.stateSpaceType = 'Continuous'

        try:
            self.stateSpaceShape = gymInst.reset().shape
            if(len(self.stateSpaceShape)==0):
                self.stateSpaceShape = (1,)
                self.stateSpaceN = self.gymInstance.observation_space.n
                self.stateSpaceType='Discrete'
        except AttributeError:
            self.stateSpaceShape = (1,)
            self.stateSpaceN = self.gymInstance.observation_space.n
            self.stateSpaceType='Discrete'

        try:
            self.actionSpace = gymInst.action_space
            self.actionSpaceShape = self.actionSpace.shape
            self.actionSpaceType='Continuous'
            if(len(self.actionSpaceShape)==0):
                self.actionSpaceShape=(1,)
                self.actionSpaceN=self.actionSpace.n
                self.actionSpaceType='Discrete'
        except:
            self.actionSpaceShape=(1,)
            self.actionSpaceN=self.actionSpace.n
            self.actionSpaceType='Discrete'

        print("**********GymSimulator is set,")
        print("*******stateSpaceShape",self.stateSpaceShape)
        print("*******actionSpace",self.actionSpace)
        print("*******actionSpaceShape",self.actionSpaceShape)
        print("*******actionSpaceType",self.actionSpaceType)


    def visualize(self, agent, nRuns=50, maxSteps=1000, verbose=1):
        """
            Draws Gym render

            agent -- object implementing method SampleActionDiscrete or SampleActionContinuous
            int nRuns -- number of rounds agent needs to play
            int maxSteps -- maximum length of a round
            int verbose -- verbosity of console output, the log is printed each verbose-th episode
        """
        allRewards = []

        for episode in range(nRuns):
            state = self.gymInstance.reset()
            rewards = []

            for step in range(maxSteps):
                self.gymInstance.render()

                if(self.actionSpaceType=="Discrete"):
                    action, _ = self.SampleActionDiscrete(state)
                else:
                    action, _ = self.SampleActionContinuous(state)
                    
                newState, reward, done, _ = env.step(action)
                rewards.append(reward)

                if done:
                    allRewards.append(np.sum(rewards))
                    if episode % verbose == 0:
                        print("episode: {}, reward: {}, average_reward: {}\n".format(episode,np.round(np.sum(rewards), decimals = 3), np.round(np.mean(allRewards[-10:]), decimals = 3)))
                    break

                state = newState


    def GetEnvState(self):
        return self.gymInstance.state


    ######RESETS the environment to given state
    def ResetFromS0(self, s0):
        """
            Makes a reset with initial state s0
        """
        self.gymInstance.reset()
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0


    def ResetFromStateSampler(self, stateSampler):
        """
            Makes a reset with initial state sampled from stateSampler
        """
        self.gymInstance.reset()
        s0=stateSampler.sample()[0,:]
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0


    ######Transition Samplers(Sequential and Parallel)
    def SampleTransitionsFromS0(self, s0, policy, NNested=1, returnRewards=False):
        """ 
                Samples transitions from state s0

                [batch,stateSpaceShape] s0 -- initial state (batch of them)
                policy -- a function state --> action, may include sampler from distribution
                NNested -- number of samples from each of states in s0
                returnRewards -- whether the method should return ONLY rewards or not
                returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], [batch,NNested,1]
                or (if returnRewards)
                returns [batch,NNested,1]
        """
        if (not len(s0.shape) == 2):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not valid, only [batch,StateSpaceShape] is acceptable")
        if (not s0.shape[1] == self.stateSpaceShape[0]):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested,self.stateSpaceShape[0]])
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,:])
                action = policy(s0[s0Id,:])
                state,reward,done,_ = self.gymInstance.step(action)

                if(not returnRewards):
                    states[s0Id,nestedId,:] = state
                    actions[s0Id,nestedId,:] = action
                    dones[s0Id,nestedId,0] = done
                rewards[s0Id,nestedId,:] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards,dones
        return rewards


    def SampleTransitionsFromStateSampler(self, stateSampler, policy, Ns0=1, NNested=1, returnRewards=False):
        """ 
            Samples transitions from state s0 ~ stateSampler

            stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
            policy -- a function state --> action, may include sampler from distribution
            int Ns0 -- number of states in s0, from which it is needed to sample transitions
            int NNested -- number of samples from each of states in s0
            bool returnRewards -- whether the method should return ONLY rewards or not
            returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], 
                [batch,NNested,1], [batch,NNested,1]
            or (if returnRewards)
                returns [batch,NNested,1]
        """
        s0 = stateSampler.sample(Ns0)
        if (not len(s0.shape) == 2):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not valid, only [batch,StateSpaceShape] is acceptable")
        if (not s0.shape[1] == self.stateSpaceShape[0]):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested,self.stateSpaceShape[0]])
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,:])
                action = policy(s0[s0Id,:])
                state,reward,done,_ = self.gymInstance.step(action)

                if(not returnRewards):
                    states[s0Id,nestedId,:] = state
                    actions[s0Id,nestedId,:] = action
                    dones[s0Id,nestedId,0] = done    
                rewards[s0Id,nestedId,0] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards, dones
        return rewards


    ######TRAJECTORY samplers(Sequential and Parallel), to sample trajectories
    def SampleTrajectoriesFromS0(self, s0, policy, returnRewards=False, returnLogProbs=False, maxIterations=400):
        """ 
                Samples Ntrajs rajectories from given state(s) s0

                [batch,stateSpaceShape] s0 -- initial states (batch of them)
                policy -- a function state --> action, may include sampler from distribution
                    policy may return (action,log_prob) if returnLogProbs is True
                bool returnRewards -- whether the method should return ONLY rewards or not
                bool returnLogProbs -- whether the method should return log_probs together with actions
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations
                returns [batch,stateSpaceShape,time],[batch,actionSpaceShape,time-1],[batch,1,time-1], [batch]
                    this is states, actions, rewards, trajectoryLengths; 
                    time is a variable dimension: length of the longest trajectory
                or (if returnRewards)
                returns [batch,1,time(list dimension?)]

                if returnLogProbs ,
                    returns additionally (list of lists) logProbs [batch,time] from policy
        """
        policyRes=policy(s0[0,:])
        if(returnLogProbs):
            if( type(policyRes) is tuple ):
                if(len(policyRes)<2 ):
                    raise PolicyMismatchError("","if returnLogProbs==True, policy should return action,log_prob")
            if(returnRewards):
                raise RegimeMismatchError("","returnLogProbs=True and returnRewards=True simultaneously is not supported")

            if(len(policyRes[0].shape)==0):
                expandActionDim=True

        else:
            if ( type(policyRes) is tuple  ):
                raise PolicyMismatchError("","if returnLogProbs==False, policy should return action")
            
            if(len(policyRes.shape)==0):
                expandActionDim=True

        if(not returnRewards):
            states  = [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]

        if(returnLogProbs):
            log_probs = [0]*s0.shape[0]

        rewards = [0]*s0.shape[0]
        
        maxLen=0
        expandActionDim=False
                
        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,:])
            state=self.GetEnvState()
            if(not returnRewards):
                states[trajId] = [state]
                actions[trajId] = []
            rewards[trajId] = []
            if(returnLogProbs):
                log_probs[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):
                policyRes=policy(state)
                if(returnLogProbs):
                    action,log_prob = policyRes
                else:
                    action=policyRes

                state,reward,done,_=self.gymInstance.step(action)
                
                #SAVING THE DATA
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]

                if(returnLogProbs):
                    log_probs[trajId] =  log_probs[trajId] + [log_prob]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1
                
            if(len(rewards[trajId])+1>maxLen):
                maxLen = len(rewards[trajId])+1
            
        #wrap np.array around them with zero padding 
        if(not returnRewards):
            statesNP = np.zeros([s0.shape[0],s0.shape[1],maxLen])
            actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
            trajLens = np.zeros([s0.shape[0]])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])

        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,:,:len(states[k])] = np.array(states[k]).transpose()        
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
                trajLens[k]=len(states[k])


            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        if(not returnRewards):
            if(returnLogProbs):
                return statesNP,actionsNP,rewardsNP, trajLens, log_probs
            else:
                return statesNP,actionsNP,rewardsNP, trajLens
        else:
            return rewardsNP


    def SampleTrajectoriesFromStateSampler(self, stateSampler, policy, Ns0=1,\
                                             returnLogProbs=False,returnRewards=False, maxIterations=400):
        """ 
                Samples Ns0 trajectories from stateSampler

                stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
                policy -- a function state --> action, may include sampler from distribution
                    policy may return (action,log_prob) if returnLogProbs is True
                Ns0 -- number of trajectories to sample

                bool returnLogProbs -- whether the method should return log_probs together with actions
                bool returnRewards -- whether the method should return ONLY rewards or not
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations
                returns [Ns0,stateSpaceShape,time],[Ns0,actionSpaceShape,time-1],[Ns0,1,time-1], [Ns0]
                    this is states, actions, rewards, trajectoryLengths; 
                    time is a variable dimension: length of the longest trajectory
                or (if returnRewards)
                returns [Ns0,1,time(list dimension?)]

                if returnLogProbs , returns additionally (list of lists) logProbs [batch,time] from policy
        """
        s0 = stateSampler.sample(Ns0)
        
        policyRes=policy(s0[0,:])
        if(returnLogProbs):
            if( type(policyRes) is tuple ):
                if(len(policyRes)<2 ):
                    raise PolicyMismatchError("","if returnLogProbs==True, policy should return action,log_prob")
            if(returnRewards):
                raise RegimeMismatchError("","returnLogProbs=True and returnRewards=True simultaneously is not supported")

            if(len(policyRes[0].shape)==0):
                expandActionDim=True
        else:
            if ( type(policyRes) is tuple  ):
                raise PolicyMismatchError("","if returnLogProbs==False, policy should return action")
            if(len(policyRes.shape)==0):
                expandActionDim=True

        if(not returnRewards):
            states  = [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]

        if(returnLogProbs):
            log_probs = [0]*s0.shape[0]

        rewards = [0]*s0.shape[0]
        
        maxLen=0
        expandActionDim=False

        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,:])
            state=self.GetEnvState()
            if(not returnRewards):
                states[trajId] = [state]
                actions[trajId] = []
            rewards[trajId] = []
            if(returnLogProbs):
                log_probs[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):
                policyRes=policy(state)
                if(returnLogProbs):
                    action,log_prob = policyRes
                else:
                    action=policyRes
                
                state,reward,done,_=self.gymInstance.step(action)
                
                #SAVING THE DATA
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]
                
                if(returnLogProbs):
                    log_probs[trajId] =  log_probs[trajId] + [log_prob]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1

            if(len(rewards[trajId])+1>maxLen):
                maxLen = len(rewards[trajId])+1
           
        #wrap np.array around them with zero padding 
        if(not returnRewards):
            statesNP = np.zeros([s0.shape[0],s0.shape[1],maxLen])
            actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
            trajLens = np.zeros([s0.shape[0]])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])

        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,:,:len(states[k])] = np.array(states[k]).transpose()        
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
                trajLens[k]=len(states[k])

            

            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        if(not returnRewards):
            if(returnLogProbs):
                return statesNP,actionsNP,rewardsNP, trajLens, log_probs
            else:
                return statesNP,actionsNP,rewardsNP, trajLens
        else:
            return rewardsNP

  
    ######TRAJECTORY samplers(Parallel), to sample trajectories
    def SampleTrajectoriesFromS0Parallel(self, pool,s0, policy, maxIterations=400):
        """ 
                Samples trajectories from given state(s) s0

                pool -- mp.pool computational unit
                [batch,stateSpaceShape] s0 -- initial state (batch of them)
                policy -- a function state --> action, may include sampler from distribution
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations
                returns [batch,stateSpaceShape,time],[batch,actionSpaceShape,time-1],[batch,1,time-1], [batch]
                    this is states, actions, rewards, trajectoryLengths; 
                    time is a variable dimension: length of the longest trajectory
        """
                                                                
        #set different random seeds for each computational thread
        seeds = (1+np.arange(s0.shape[0]))*1035
        #call parallel code
        trajs = pool.starmap(self.SampleTrajectoriesFromS0ParallelOne, \
                         [(s0[k,:],policy,maxIterations,seeds[k]) for k in np.arange(s0.shape[0])]  )

        #zero padding
        #wrap np.array around them with zero padding 
        #compute maxLen
        maxLen = np.amax([len(trajs[k][0]) for k in np.arange(len(trajs))])
        statesNP = np.zeros([len(trajs),s0.shape[1],maxLen])
        actionsNP = np.zeros([len(trajs),self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([len(trajs),1,maxLen-1])
        trajLens = np.zeros([len(trajs)])
        for k in np.arange(len(trajs)):
            statesNP[k,:,:len(trajs[k][0])] = np.array(trajs[k][0]).transpose()        
            actionsNP[k,:,:len(trajs[k][1])] = np.array(trajs[k][1]).transpose()
            
            rewardsNP[k,0,:len(trajs[k][2])] = np.array(trajs[k][2])
            trajLens[k]=len(trajs[k][0])

        return statesNP,actionsNP,rewardsNP, trajLens                



    def SampleTrajectoriesFromS0ParallelOne(self,s0,policy,maxIterations,seed=1002312):
        """  
                Handler for parallel sampling (above)

                [stateSpaceShape] s0 -- initial state
                policy -- a function state --> action, may include sampler from distribution
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations
                int seed -- random seed
                returns [list(time),stateSpaceShape],[list(time-1),actionSpaceShape],[list(time-1),1]
                this is states, actions, rewards; 
                time is a variable dimension: length of the longest trajectory
        """
        np.random.seed(seed)
        states =  [] # lists of lists
        actions = []
        rewards = []
        
        maxLen=0
        expandActionDim=False
        
        if(len(policy(s0).shape)==0):
            expandActionDim=True
        
        self.ResetFromS0(s0)
        states.append(self.GetEnvState())

        iterId=0
        done=False
        while(iterId<maxIterations and (not done)):
               
            action = policy(states[iterId])
            state,reward,done,_=self.gymInstance.step(action)
            
            if(expandActionDim):
                action=np.array([action])

            states.append(state)
            actions.append(action)

            rewards.append(reward)
            iterId = iterId + 1
            
        return states,actions,rewards


    def SampleTrajectoriesFromStateSamplerParallel(self, pool,stateSampler, policy, Ns0=1, maxIterations=400):
        """ 
                Samples Ns0 trajectories from given stateSampler

                pool -- mp.pool object for computations
                stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
                policy -- a function state --> action, may include sampler from distribution
                Ns0 -- number of trajectories to sample
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations
                returns [Ns0,stateSpaceShape,time], [Ns0,actionSpaceShape,time], [Ns0,1,time], [Ns0]
                this is states, actions, rewards, trajectoryLengths
        """
        #sample initial states from stateSampler
        s0 = stateSampler.sample(Ns0)

        #set different random seed for each computational thread
        seeds = (1+np.arange(s0.shape[0]))*1035 #

        #run parallel code
        trajs = pool.starmap(self.SampleTrajectoriesFromS0ParallelOne, \
                         [(s0[k,:],policy,maxIterations,seeds[k]) for k in np.arange(s0.shape[0])]  )

        #zero padding
        #wrap np.array around them with zero padding 
        #compute maxLen
        maxLen = np.amax([len(trajs[k][0]) for k in np.arange(len(trajs))])
        statesNP = np.zeros([len(trajs),s0.shape[1],maxLen])
        actionsNP = np.zeros([len(trajs),self.actionSpaceShape[0],maxLen-1])
        rewardsNP = np.zeros([len(trajs),1,maxLen-1])
        trajLens = np.zeros([len(trajs)])
        for k in np.arange(len(trajs)):
            statesNP[k,:,:len(trajs[k][0])] = np.array(trajs[k][0]).transpose()        
            actionsNP[k,:,:len(trajs[k][1])] = np.array(trajs[k][1]).transpose()
            
            rewardsNP[k,0,:len(trajs[k][2])] = np.array(trajs[k][2])
            trajLens[k]=len(trajs[k][0])

        return statesNP,actionsNP,rewardsNP, trajLens



##################################### TENSOR STATE ###############################################

class GymTensorStateSimulator:
    #GymSimulator for the environments which represent state as tensor (ATARI, image-based...)

    def __init__(self,gymInst):
        """
            Initializes GymTensorStateSimulator

            gymInstance -- gym environment
        """

        self.gymInstance = gymInst

        #by defaut it is assumed  to be tensor
        self.stateSpaceShape = gymInst.reset().shape        

        self.actionSpace = gymInst.action_space
        self.actionSpaceShape = self.actionSpace.shape
        self.actionSpaceType='Continuous'

        if(len(self.actionSpaceShape)==0):
            self.actionSpaceShape=(1,)
            self.actionSpaceN=self.actionSpace.n
            self.actionSpaceType='Discrete'

        print("**********GymTensorStateSimulator is set,")
        print("*******stateSpaceShape",self.stateSpaceShape)
        print("*******actionSpace",self.actionSpace)
        print("*******actionSpaceShape",self.actionSpaceShape)
        print("*******actionSpaceType",self.actionSpaceType)


    def GetEnvState(self):
        return self.gymInstance.state

    ######RESETS the environment to given state
    def ResetFromS0(self, s0):
        self.gymInstance.reset()
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0


    def ResetFromStateSampler(self, stateSampler):
        self.gymInstance.reset()
        s0=stateSampler.sample()[0,...]
        if(not self.stateSpaceShape == s0.shape):
            raise ShapeMismatchError("", "The stateSpaceShape "+str(self.stateSpaceShape) +" is not the same as s0's "+str(s0.shape))
            return -1
        self.gymInstance.state=s0
        return s0


    ######Transition Samplers(Sequential and Parallel)
    def SampleTransitionsFromS0(self, s0, policy, NNested=1, returnRewards=False):
        """ 
                Samples transitions from state s0

                s0 -- [batch,stateSpaceShape]
                policy -- a function state --> action, may include sampler from distribution
                NNested -- number of samples from each of states in s0
                returnRewards -- whether the method should return ONLY rewards or not
                returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], [batch,NNested,1]
                or (if returnRewards)
                returns [batch,NNested,1]
        """
        if (not np.all( np.array(list(s0.shape))[1:] == np.array(list(self.stateSpaceShape)) ) ):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested]+list(self.stateSpaceShape) )
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,...])
                action = policy(s0[s0Id,...])
                state,reward,done,_ = self.gymInstance.step(action)

                if(not returnRewards):
                    states[s0Id,nestedId,...] = state
                    actions[s0Id,nestedId,...] = action
                    dones[s0Id,nestedId,0] = done
                rewards[s0Id,nestedId,...] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards,dones
        return rewards


    def SampleTransitionsFromStateSampler(self, stateSampler, policy, Ns0=1, NNested=1, returnRewards=False):
        """ 
            Samples transitions from state s0 ~ stateSampler

            stateSampler -- object implementing method sample(Nsamples) returning [Nsamples,stateSpaceShape]
            policy -- a function state --> action, may include sampler from distribution
            int Ns0 -- number of states in s0, from which it is needed to sample transitions
            int NNested -- number of samples from each of states in s0
            bool returnRewards -- whether the method should return ONLY rewards or not

            returns [batch,NNested,stateSpaceShape], [batch,NNested,actionSpaceShape], 
                [batch,NNested,1], [batch,NNested,1]
            or (if returnRewards)
                returns [batch,NNested,1]
        """
        s0 = stateSampler.sample(Ns0)
        if (not np.all( np.array(list(s0.shape))[1:] == np.array(list(self.stateSpaceShape)) ) ):
            raise ShapeMismatchError("","The s0.shape "+str(s0.shape) + " is not compatible with StateSpaceShape="+str(self.stateSpaceShape) )

        if(not returnRewards):
            states = np.zeros([s0.shape[0],NNested] +list(self.stateSpaceShape) )
            actions = np.zeros([s0.shape[0],NNested,self.actionSpaceShape[0]])
            dones = np.zeros([s0.shape[0],NNested,1]).astype("bool")
        rewards = np.zeros([s0.shape[0],NNested,1])

        for s0Id in np.arange(s0.shape[0]):
            for nestedId in np.arange(NNested):
                self.ResetFromS0(s0[s0Id,...])
                action = policy(s0[s0Id,...])
                state,reward,done,_ = self.gymInstance.step(action)

                if(not returnRewards):
                    states[s0Id,nestedId,...] = state
                    actions[s0Id,nestedId,...] = action
                    dones[s0Id,nestedId,0] = done    
                rewards[s0Id,nestedId,...] = reward                
            
        if(not returnRewards):            
            return states,actions,rewards, dones
        return rewards


    ######TRAJECTORY samplers, to sample trajectories

    def SampleTrajectoriesFromS0(self, s0, policy, returnRewards=False, returnLogProbs=False, maxIterations=400):
        """ 
                Samples Ntrajs rajectories from given state(s) s0
                [batch,stateSpaceShape] s0 -- initial states (batch of them)
                policy -- a function state --> action, may include sampler from distribution
                    policy may return (action,log_prob) if returnLogProbs is True
                bool returnRewards -- whether the method should return ONLY rewards or not
                bool returnLogProbs -- whether the method should return log_probs together with actions
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations

                returns [batch,stateSpaceShape,time],[batch,actionSpaceShape,time-1],[batch,1,time-1], [batch]
                    this is states, actions, rewards, trajectoryLengths; 
                    time is a variable dimension: length of the longest trajectory
                or (if returnRewards)
                    returns [batch,1,time(list dimension?)]

                if returnLogProbs ,
                    returns additionally (list of lists) logProbs [batch,time] from policy
        """
        policyRes=policy(s0[0,...])
        if(returnLogProbs):
            if( type(policyRes) is tuple ):
                if(len(policyRes)<2 ):
                    raise PolicyMismatchError("","if returnLogProbs==True, policy should return action,log_prob")
            if(returnRewards):
                raise RegimeMismatchError("","returnLogProbs=True and returnRewards=True simultaneously is not supported")
            if(len(policyRes[0].shape)==0):
                expandActionDim=True
        else:
            if ( type(policyRes) is tuple  ):
                raise PolicyMismatchError("","if returnLogProbs==False, policy should return action")
            
            if(len(policyRes.shape)==0):
                expandActionDim=True

        if(not returnRewards):
            states  = [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]

        if(returnLogProbs):
            log_probs = [0]*s0.shape[0]

        rewards = [0]*s0.shape[0]
        
        maxLen=0
        expandActionDim=False
        
        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,...])
            state=self.GetEnvState()
            if(not returnRewards):
                states[trajId] = [state]
                actions[trajId] = []
            rewards[trajId] = []
            if(returnLogProbs):
                log_probs[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):
                policyRes=policy(state)
                if(returnLogProbs):
                    action,log_prob = policyRes
                else:
                    action=policyRes

                state,reward,done,_=self.gymInstance.step(action)
                
                #SAVING THE DATA
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]

                if(returnLogProbs):
                    log_probs[trajId] =  log_probs[trajId] + [log_prob]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1
    
            if(len(rewards[trajId])+1>maxLen):
                maxLen = len(rewards[trajId])+1
            
        #wrap np.array: state,actions,rewards with zero padding 
        if(not returnRewards):
            statesNP = np.zeros(list(s0.shape)+[maxLen])
            actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
            trajLens = np.zeros([s0.shape[0]])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])

        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,...,:len(states[k])] = np.transpose(np.array(states[k]),\
                                                     tuple([k for k in np.arange(1,len(states[k][0].shape)+1)]+[0]) )
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
                trajLens[k]=len(states[k])

            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        if(not returnRewards):
            if(returnLogProbs):
                return statesNP,actionsNP,rewardsNP, trajLens, log_probs
            else:
                return statesNP,actionsNP,rewardsNP, trajLens
        else:
            return rewardsNP


    #preprocessing routines for graphic input
    def to_grayscale(self, img):
        return np.mean(img, axis=-1, keepdims=True).astype(np.uint8)

    def downsample(self, img):
        return img[..., ::2, ::2, :]

    def preprocess(self, img):
        return self.to_grayscale(self.downsample(img))

    def SampleTrajectoriesFromStateSampler(self, stateSampler, policy, Ns0=1,\
                                             returnLogProbs=False,returnRewards=False, maxIterations=400, stateMemorySize=0, grayscale=False, downsample=False):
        """ 
                Samples Ns0 trajectories from stateSampler

                stateSampler -- an object which implements method sample(Nsamples) returning [Nsamples,stateSpaceShape]
                policy -- a function state --> action, may include sampler from distribution
                    policy may return (action,log_prob) if returnLogProbs is True
                int Ns0 -- number of trajectories to sample

                bool returnLogProbs -- whether the method should return log_probs together with actions
                bool returnRewards -- whether the method should return ONLY rewards or not
                int maxIterations -- iteration ceiling for simulation: no trajectory will be sampled for more that maxIterations iterations

                int stateMemorySize -- amount of additional frames representing a single state (for graphic state)
                bool grayscale -- convert multilayered images to gray image (for graphic state)
                bool downsample -- lower the resolution by picking every second pixel (for graphic state)

                returns [Ns0,stateSpaceShape,time],[Ns0,actionSpaceShape,time-1],[Ns0,1,time-1], [Ns0]
                    this is states, actions, rewards, trajectoryLengths; 
                    time is a variable dimension: normalised w.r.to length of the longest trajectory
                or (if returnRewards)
                    returns [Ns0,1,time-1]

                if returnLogProbs ,
                    returns additionally (list of lists) logProbs [batch,time] from policy
        """

        s0 = stateSampler.sample(Ns0)
        
        if stateMemorySize > 0:
            if grayscale == True:
                s0_t = self.to_grayscale(s0[0,...])
            else:
                s0_t = s0[0,...]
            for _ in range(stateMemorySize):
                if grayscale == True:
                    s0_t = np.concatenate((s0_t, self.to_grayscale(s0[0,...])), axis=-1)
                else:
                    s0_t = np.concatenate((s0_t, s0[0,...]), axis=-1)
            if downsample == True:
                s0_t = self.downsample(s0_t)
            policyRes=policy(s0_t)
        else:
            policyRes=policy(s0[0,...])
        if(returnLogProbs):
            if( type(policyRes) is tuple ):
                if(len(policyRes)<2 ):
                    raise PolicyMismatchError("","if returnLogProbs==True, policy should return action,log_prob")
            if(returnRewards):
                raise RegimeMismatchError("","returnLogProbs=True and returnRewards=True simultaneously is not supported")

            if(len(policyRes[0].shape)==0):
                expandActionDim=True
        else:
            if ( type(policyRes) is tuple  ):
                raise PolicyMismatchError("","if returnLogProbs==False, policy should return action")
            if(len(policyRes.shape)==0):
                expandActionDim=True

        if(not returnRewards):
            states  = [0]*s0.shape[0] # lists of lists
            actions = [0]*s0.shape[0]

        if(returnLogProbs):
            log_probs = [0]*s0.shape[0]

        rewards = [0]*s0.shape[0]
        
        maxLen=0
        expandActionDim=False
        
        input_list = list(self.stateSpaceShape[:-1])
        if downsample == True:
            input_list[-1] //= 2
            input_list[-2] //= 2
        input_dims = tuple(input_list)
        
        
        for trajId in np.arange(s0.shape[0]):
            self.ResetFromS0(s0[trajId,...])
            state=self.GetEnvState()
            if grayscale == True:  
                state = self.to_grayscale(state)
            if downsample == True:
                state = self.downsample(state)
            if(not returnRewards):
                states[trajId] = [state]
                actions[trajId] = []
            rewards[trajId] = []
            if(returnLogProbs):
                log_probs[trajId] = []
            
            iterId=0
            done=False
            while(iterId<maxIterations and (not done)):

                if iterId<stateMemorySize:
                    if(returnLogProbs):
                        action,log_prob = 0, 0
                    else:
                        action = 0
                    state,reward,done,_=self.gymInstance.step(action)
                else:
                    if stateMemorySize > 0:
                        state_step = states[trajId][-stateMemorySize:] + [state]
                        state_step = np.array(state_step).reshape((*input_dims, -1))
                        policyRes=policy(state_step)
                    else:    
                        policyRes=policy(state)
                    if(returnLogProbs):
                        action,log_prob = policyRes
                    else:
                        action=policyRes
                    
                    state,reward,done,_=self.gymInstance.step(action)

                if grayscale == True:  
                    state = self.to_grayscale(state)
                if downsample == True:
                    state = self.downsample(state)

                #SAVING THE DATA
                if(expandActionDim):
                    action=np.array([action])
                if(not returnRewards):
                    states[trajId] = states[trajId] + [state]
                    actions[trajId] = actions[trajId] + [action]
                    
                if(returnLogProbs):
                    log_probs[trajId] =  log_probs[trajId] + [log_prob]

                rewards[trajId] = rewards[trajId] + [reward]
                iterId = iterId + 1

            if(len(rewards[trajId])+1>maxLen):
                maxLen = len(rewards[trajId])+1
           
        #wrap np.array: states,actions and rewards with zero padding 
        if(not returnRewards):
            s0shape = list(s0.shape)
            if grayscale == True:
                s0shape[-1] = 1
            if downsample == True:
                s0shape[-2] //= 2
                s0shape[-3] //= 2
            statesNP = np.zeros(s0shape+[maxLen])
            actionsNP = np.zeros([s0.shape[0],self.actionSpaceShape[0],maxLen-1])
            trajLens = np.zeros([s0.shape[0]])
        rewardsNP = np.zeros([s0.shape[0],1,maxLen-1])

        for k in np.arange(s0.shape[0]):
            if(not returnRewards):
                statesNP[k,...,:len(states[k])] = np.transpose(np.array(states[k]),\
                                                     tuple([k for k in np.arange(1,len(states[k][0].shape)+1)]+[0]) )
                actionsNP[k,:,:len(actions[k])] = np.array(actions[k]).transpose()
                trajLens[k]=len(states[k])

            rewardsNP[k,0,:len(rewards[k])] = np.array(rewards[k])

        if(not returnRewards):
            if(returnLogProbs):
                return statesNP,actionsNP,rewardsNP, trajLens, log_probs
            else:
                return statesNP,actionsNP,rewardsNP, trajLens
        else:
            return rewardsNP
