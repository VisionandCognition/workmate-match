#!/usr/bin/env python
"""

This file is adapted from  the WorkMATe source code, provided as supplemental material
for the article:
    "Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers

-------------------
This file implements the Delayed Match-to-Sample task used as a learning environment
These classes at least implement:
    - __init__()    A constructor that at least defines the n_actions 
                    for the task
    - reset()       A function to initialize a new trial 
    - step(action)  A function that evaluates the agent's action, and returns 
                    a new observation and a reward in return.
                    if this observation is 'RESET', the agent assumes the trial
                    has ended and will reset its tags and memory content

For testing purposes, this file also implements a 'Manual_agent', which allows
one to take the role of a WorkMATe agent and reply to trial sequences  with
keyboard input

"""
#%% Dependencies

import numpy as np 
import string

#%% Task environment
class DMS(object):
    """Delayed Match to sample, with switches in stimset"""
    def __init__(self, n_stim=3, n_switches=4, stim_type = 'alpha'):
        super(DMS, self).__init__()
        self.n_actions = 3 # hold fixation, go left/mismatch or go right/match
        self.stim_type = stim_type #type of stimulus that is being used in the task
        self.n_stim = n_stim
        self.n_switches = n_switches
        
        self.minireward = 0.2 #0.2
        self.bigreward  = 1.5 #1.5
        self.noreward = 0

        # DETERMINE NEW STIMULI
        n_blocks = n_switches + 1
        #For the letter stimuli
        if self.stim_type == 'alpha':
            self.fix = 'p' #stimulus for fixation point
            stims = np.random.choice(list(string.ascii_uppercase), n_stim * n_blocks, replace=False)
            
        #For the cifar-10 latent vector representations
        if self.stim_type == 'cifar':
            self.fix = 359 #representation with the smallest correlation
            #see inputs.get_fix_stim()
            stim_num = 400
            #stim_num = 10000 #number of stims used to choose from
            stims = np.random.choice([i for i in range(0,stim_num) if i != self.fix], n_stim * n_blocks, replace=False)
        
        self.STIM = stims
        self.stimset = self.STIM.reshape( (n_switches+1, n_stim) )
        self.setnr   = 0 
        self.stimuli = self.stimset[self.setnr, :]
        self._newstimuli = stims[-n_stim:] #what is this?
        self.reset()
        return
 
    def reset(self):
        self.trial_type = np.random.choice([0,1]) #0 = match
        #print(self.trial_type)
        self.target = self.trial_type * 2
        sample, probe = np.random.choice(self.stimuli, 2, replace=False)
        if self.trial_type == 0:
            probe = sample
        
        self.seq = [self.fix, sample, self.fix, probe] #more general code for numbers also
        #self.seq = list("p{}p{}".format(sample, probe))
              
        self.t = 0
        return

    def switch_set(self, target_set=None):
        # Cycle through new- and old sets:
        if target_set is None:
            target_set = (self.setnr + 1) % self.stimset.shape[0]
        assert target_set < self.stimset.shape[0] # assert always new, unknown
        self.stimuli = self.stimset[target_set, :]
        self.setnr = target_set
        return

    def step(self,action):
        self.t += 1 #no action is taken at t = 0
        if action == -1:
            self.reset()
            newobs = self.seq[ self.t ]
            #self.t += 1
            return newobs, 0.0
        # else:
        if action == 1:
            newobs, reward = self._hold()
            #self.t += 1
        else:
            newobs, reward = self._go(action)
            #self.t += 1
        return newobs, reward

    def _hold(self):
        try:
            newobs =  self.seq[ self.t ]
            return newobs, self.minireward
        except IndexError as e:
            # out of time, should've gone!
            pass
        return  'RESET', 0.0

    def _go(self, action):
        # is it the end of the sequence?
        if self.t < len(self.seq):
            return 'RESET', 0.0   # if it is not the end of the sequence, it will be resetted?
        # was it on the target?
        elif action == self.target:
            return 'RESET', self.bigreward
        return 'RESET', -1*self.minireward #this is a kind of supervised learning?? 

#%% Define agent that can be used to step-wise explore the environment

class Manual_agent(object):
    """for easy testing of env objects"""
    def __init__(self, env):
        super(Manual_agent, self).__init__()
        self.env = env
        self.action = -1
        self.t = 0
        return

    def step(self):
        obs, r = self.env.step(self.action)
        print("OBSERVATION {}:\t{}".format(self.t, obs))
        print("REWARD\t\t{}".format(r))
        if obs == 'RESET':
            self.action = -1
            self.t = 0
            print("-----")
            print("NEW TRIAL")
            print("-----")
            return 

        print("Choose an action: {}".format(list(range(self.env.n_actions))))
        act =  eval(input('? > '))
        self.action = int(act)
        self.t += 1
    
        return

# %% Run manual agent
if __name__ == '__main__':
    env = DMS()
    agent = Manual_agent(env)
    env.reset()
    aa = True                   #take action 5 to break out of the loop!
    while aa:
        aa = agent.step()
        agent.step()
