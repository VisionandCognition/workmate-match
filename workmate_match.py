# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:59:09 2021

This file is is and adaptation of the WorkMATe source code, from the article:
    "Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers

-------------------
The class defined in this file implements the WorkMATe model;
Entry point for the model is the 'step()' function which defined what WorkMATe
does during a single time step; this entails: 
1-  the feedforward sweep to determine its  next action, 
2-  Learning: update the weights in response to any potentially obtained reward
3-  recurrent attentional feedback to place synaptic tags on connections 
    responsible for the selected action. 
4-  Executing the desired action, including possibly storing 
    an observation in memory.

Several extensions are added to the original code --> making it a Deep WorkMATe
with CIFAAR-10 inputs and a locally learned matching function. 
1- An extra layer was added (latent) 
2- Timestamp is added at latent level, learning rate is set to 0.05
3- Only weight updates are performed between trials
4- Both CIFAR-10 latent features and the alphabet can be used as a stimulus dictionary

@author: Lieke Ceton
"""
#%% Dependencies

import numpy as np 

from stimuli import obs_time, obs_cifar, match_cifar, obs_alpha, match_alpha
from matchnet import MatchNet, cosine_similarity

#%% Define WorkMATe architecture

#np.random.seed(20)
class WorkMATe(object):
    """
    Architecture schematic:
    x -- l -- S
          \/
          h
          |
    [q_int|q_ext]
    """
    def __init__(self, env=None, nhidden=25, nblocks=2, block_size=20, lr=0.05, wh=0.5):
        super(WorkMATe, self).__init__()
        
        #np.random.seed(20)
        assert env is not None
        self.env=env
        
        ## learning params (adopted from Rombouts et al., 2015)
        self.beta = lr
        self.gamma = 0.90
        self.L = 0.8
        # exploration:
        self.epsilon = 0.025
        self.bias = 1
        print('Agent WorkMATe_match initialising')
        
        ## member lambda functions:
        # sigmoid transfer function, offset at 2.5
        sigmoid_offset = 2.5
        self.transfer = lambda x: 1 / ( 1. + np.exp(sigmoid_offset - x) )
        self.dtransfer= lambda x: x * (1. - x) # derivative
        
        # softmax normalization; for action selection - boltzmann controller
        self.softmaxnorm = lambda x: (
            np.exp( x - x.max() ) / np.exp( x - x.max() ) .sum() )

        ## init network architecture -- inputs and output shape from env
        # input and hidden
        #nx = inputs.get_obs('a').size
        nl = block_size
        nh = nhidden
        # memory cell properties:
        self.nblocks = nblocks
        self.block_size = block_size
        nS = nblocks * block_size
        
        # output -- q layer consisting of 2 modules
        # module for n external actions, internal actions for nblocks + 1 (null) 
        mod_sz = env.n_actions, nblocks + 1
        nq = np.sum(mod_sz)
        # indices of module for each node:
        self.zmods = np.hstack( [ [i] * sz for i, sz in enumerate( mod_sz ) ] )

        ## init network layers (activations 0)
        # (x will be constructed when processing 'new_obs')
        time_input = obs_time().size
        if self.env.stim_type == 'alpha':
            letter_input = obs_alpha().size
        if self.env.stim_type == 'cifar':
            letter_input = obs_cifar().size
        
        self.S = np.zeros( nS )
        self.l = np.zeros( nl )
        self.h = np.zeros( nh ) 
        self.q = np.zeros( nq ) 

        ## init weights, tags traces, (+1 indicates projection from bias node)
        wl, wh = -wh, wh
        
        #Characteristics used as attributes for step_match()
        self.ch = [letter_input, nl, nh, nS, wh, wl]  
        
        # Memory projection (x > S)
        #self.W_Sx  = np.random.sample( (nS, nx) )  * (wh-wl) + wl 
        self.W_Sl  = np.random.sample( (nS, nl) )  * (wh-wl) + wl 
        # Note that time and sensory input cells are not separated in memory

        # PLASTIC CONNECTIONS (all except memory projection)
        
        # Input projection with bias node
        self.W_lx = np.random.sample((nl, letter_input + 1))*(wh-wl) + wl
        #self.W_lx = np.array(W_lx)
        self.W_lx_start = np.copy(self.W_lx)
        
        # connections l -> h; nl + match nodes + bias
        self.W_hl  = np.random.sample( (nh, time_input + nl + nblocks + 1) ) * (wh-wl) + wl
        self.W_hl_start = np.copy(self.W_hl)

        # connections S-> h:
        self.W_hS  = np.random.sample( (nh, nS    ) ) * (wh-wl) + wl
        # connections h-> q:
        self.W_qh  = np.random.sample( (nq, nh + 1) ) * (wh-wl) + wl
        
        # tags are shaped like weights but initialized at 0:
        zeros_ = np.zeros_like
        # W_lx is only updated between trials to keep memory trace stable 
        # W_lx_trial accumulates all changes made within a trial 
        self.W_lx_trial = zeros_(self.W_lx)

        #initialise tags
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_hS, self.Trace_W_hS = zeros_(self.W_hS), zeros_(self.W_hS)
        self.Tag_W_qh, self.Trace_W_qh = zeros_(self.W_qh), zeros_(self.W_qh)
        
        ## Init matchnet:
        dummy_inp = np.zeros(block_size) #dummy input of length S == Sproj and Memory
        self.matchnet = [MatchNet(dummy_inp, dummy_inp) for item in range(self.nblocks)]
       
        # Init action state
        self.action = -1
        # (prev) predicted reward:
        self.qat_1 = self.qat = None
        self.t   = 0 
        return

    def _intertrial_reset(self):
        """
        Reset time, memory, tags and traces
        """
        # update weights W_lx with the accumulated weight updates from the whole trial 
        self.W_lx += self.W_lx_trial

        # reset time and memory
        self.t = 0 
        self.S *= 0

        # previous action = zeros 
        self.z *= 1 #I set this from 0 to 1, it does not seem to change anything
        # reset tags/traces for each Wmat
        zeros_ = np.zeros_like
        self.Tag_W_lx, self.Trace_W_lx = zeros_(self.W_lx), zeros_(self.W_lx)
        self.Tag_W_hl, self.Trace_W_hl = zeros_(self.W_hl), zeros_(self.W_hl)
        self.Tag_W_hS, self.Trace_W_hS = zeros_(self.W_hS), zeros_(self.W_hS)
        self.Tag_W_qh, self.Trace_W_qh = zeros_(self.W_qh), zeros_(self.W_qh)
        #reset accumulated weight updates W_lx
        self.W_lx_trial = zeros_(self.W_lx)
        
        # reset 'current action', and the like
        self.action = -1
        self.qat = None
        return
    
    def step_match(self):
        if self.env.stim_type == 'cifar':
            self.x_sens = match_cifar() #get random input
        elif self.env.stim_type == 'alpha':
            self.x_sens = match_alpha() #get random input
            
        self.x = np.r_[self.x_sens, self.bias]
        Sproj = self.compute_latent()   #Compute the memory projection
        Sproj = Sproj.reshape((self.nblocks, self.block_size)) 
        
        #Decide on trial type and thus on memory content
        #For each memory block there is a match network
        #tt is the trial type: 0 is match, 1 is mismatch
        #If tt = 0, the memory is updated to the current input
        #The chance is around equal to save or not
        tt = np.random.choice([0,1], self.nblocks, p=[0.5, 0.5])
        S_ = self.S.reshape( (self.nblocks, self.block_size) )
        for i in range(self.nblocks):
            if tt[i] == 0:
                S_[i,:] = Sproj[i,:]#the S projection should be here!
                            
        #Compute the match values
        m, tt, corr = self.compute_match(Sproj)
        
        #Noise/random changes in weights W_lx and W_Sl
        self.W_lx = np.random.sample((self.ch[1], self.ch[0] + 1))*(0.5--0.5) - 0.5
        self.W_Sl  = np.random.sample((self.ch[3], self.ch[1]))*(self.ch[4]-self.ch[5]) + self.ch[5]

        return m, tt, corr #return match values and trial_types
        
    def step(self):
        # get observation and reward from env
        self.obs, self.r = self.env.step(self.action) 
        # do feedforward:
        #matches, tt_match = self._feedforward()
        matches, tt = self._feedforward() 
        # learn from the obtained reward
        self._learn()
        # end of trial?
        if 'RESET' == self.obs: #change in to == because the observation is now an integer not a string
            self._intertrial_reset()
            return self.r, None, None #fake matches
        # do feedback (tag placement)
        self._feedback()
        # act (internal, external)
        self._act()
        self.t += 1
        #return self.r, matches #, tt_match
        return self.r, matches, tt

    def _feedforward(self):
        # shift previous action
        self.qat_1 = self.qat
        if 'RESET' == self.obs: #change in to == because the observation is now an integer not a string
            # however, we do not expect RESET to ever only be part of the observation so the equal to boolean should be good enough
            # no meaningful feedforward sweep:  qat is not computed
            self.qat = None
            return None, None #return fake match
        # else:
        # compute input, hidden, output:
        self.construct_input()
        Sproj = self.compute_latent()
        matches, tt, _ = self.compute_match(Sproj)
        self.compute_hidden(matches)
        self.compute_output()
        # determine z from q (action selection)
        self.action_selection()
        # determine new qat
        self.qat = (self.z * self.q).sum()
        return matches, tt

    def _learn(self):
        """
        Learn from the reward; compute RPE and update weights
        general form form delta = r + gamma * qat - qat_1
        ...but there are edge cases
        """
        r = self.r
        if self.qat and self.qat_1: # regular
            delta = r + (self.gamma * self.qat) - self.qat_1
        elif self.qat_1 is None: # first step
            delta = r + (self.gamma * self.qat) - self.qat
        else: # self.qa(t) is None (final step):
            delta = r - self.qat_1
        self.delta = delta
        self.update_weights()
        return

    def _feedback(self):
        # updates traces and tags, based on action selection:
        # traces and tags
        self.update_traces()
        self.update_tags()
        return

    def _act(self):
        # external and internal actions:
        zext =  self.z[self.zmods == 0]
        zint =  self.z[self.zmods == 1]
        self.action = np.argmax(zext)
        self.update_memory( zint )
        return

    def construct_input(self):
        """
        Turn obs into a vector; uses coding defined in 'inputs.py'
        """
        # input consists of: observation and time t
        self.x_time = obs_time(self.t) #time-part
        #letter-part
        if self.env.stim_type == 'alpha':
           self.x_sens = obs_alpha(self.obs)
           #self.x_sens = inputs.obs_alpha_with_noise(self.obs)
        if self.env.stim_type == 'cifar':
           self.x_sens = obs_cifar(self.obs)
           #self.x_sens = inputs.obs_orthogonal(self.obs)
        #self.x_sens = inputs.get_obs(self.obs, self.t) #only letter-part
        self.x = np.r_[self.x_sens, self.bias] #only bias included 
        return
    
    def compute_latent(self):
        # x -> l 
        self.l_in = self.W_lx.dot(self.x)
        # Compute l activities
        self.l_sens = self.transfer(np.r_[self.l_in]) 
        # Compute match value:
        Sproj = self.W_Sl.dot(self.l_in).reshape( (self.nblocks, self.block_size) )
        return Sproj
    
    def compute_match(self, Sproj):
        #output = self.matchnet.step(x,y)
        matches = []
        corr = []
        tt = []
        S_ = self.S.reshape( (self.nblocks, self.block_size) )
        for i in range(self.nblocks):
            m, _ = self.matchnet[i].step(Sproj[i,:], S_[i,:])
            cos_corr = cosine_similarity(Sproj[i,:], S_[i,:])
            corr.append(cos_corr)
            matches.append(m)
            if np.all(Sproj[i,:] == S_[i,:]):
                 tt_match = 0
            else: 
                 tt_match = 1
            tt.append(tt_match)
        return matches, tt, corr
    
    def compute_hidden(self, matches):
        # add match nodes + bias to input vector
        #add time-part here
        self.l_out = np.r_[self.x_time, self.l_sens, matches, self.bias] #the matches are added here now
        
        # x->h  +  S->h
        self.S_out  = self.S

        # Compute Ha and h
        h_in = self.W_hl.dot(self.l_out) + self.W_hS.dot(self.S_out)
        self.h_out  = np.r_[self.transfer(h_in), self.bias] # bias added
        return

    def compute_output(self):
        # hidden output (has bias added)
        self.q = self.W_qh.dot(self.h_out)
        # (no transfer, q nodes are linear)
        return

    def action_selection(self):
        #np.random.seed(self.seed)
        # using q, per module, determine z (based on argmax or exploration)
        self.z = np.zeros_like(self.q)
        # action selection for both modules separately
        for mod_idx in np.unique(self.zmods):
            qvec = self.q[self.zmods == mod_idx] # get the module's qvalues 
            # check exploration; if not just take argmax:
            if ( np.random.sample() >= self.epsilon ):
                action = np.argmax(qvec)
            else: # compute softmax over Q and explore:
                pvec = self.softmaxnorm(qvec)
                action = np.random.choice( list(range(qvec.size)), p = pvec) 
            # set zvec: 1-hot code of actions:            
            zvec = np.zeros_like(qvec)
            zvec[ action ] = 1.0
            # place zvec into z at the right indices:
            self.z[self.zmods == mod_idx] = zvec
        return

    def update_weights(self):
        """
        all Weight-Trace pairs are updated with the same rule:
        w += beta * delta * tag
        """
        #Only update W_lx during intertrial resets
        #Accumulate changes during trial and update later
        self.W_lx_trial += self.beta * self.delta * self.Tag_W_lx
        
        #Change the weights between h and S slower to have memory..
        self.W_hS += self.beta * self.delta * self.Tag_W_hS
        
        wt_pairs = ( 
                     ( self.W_hl, self.Tag_W_hl ),    
                     ( self.W_qh, self.Tag_W_qh )) #added by LJC
        for W, Tag in wt_pairs:
            W += self.beta * self.delta * Tag
        
        return

    def update_traces(self):
        """
        Traces are the intermediate layers' markers
        The Traces are a relic from old AuGMEnT code, have no 'meaning' here
        """
        # Regulars, are replaced by new input:
        self.Trace_W_lx *= 0.0
        self.Trace_W_hl *= 0.0
        self.Trace_W_hS *= 0.0
        # add 1 x X vec to H x X matrix yields H copies of 1 x X vec
        self.Trace_W_lx += self.x.reshape(1, self.x.size) # this includes trace for bias
        self.Trace_W_hl += self.l_out.reshape(1, self.l_out.size) # this includes trace for bias
        self.Trace_W_hS += self.S_out.reshape(1, self.S.size)
        return

    def update_tags(self):
        
        # 1. old tag decay:
        alltags = (self.Tag_W_lx, self.Tag_W_hl, self.Tag_W_hS, self.Tag_W_qh)
        for Tag in alltags:
            Tag *= (self.L * self.gamma)

        # 2. form new tags:
        # tags onto output units: selected action.
        self.Tag_W_qh[self.z.astype('bool'), :] += self.h_out

        # feedback to hidden
        dh = self.dtransfer(self.h_out[:-1])
        self.fbh = self.W_qh[self.z.astype('bool'), :-1] # excluding the bias node
        self.fbh = self.fbh.sum(axis = 0 ) # summed contribution of all actions
        self.feedbackh = self.fbh * dh
        self.Tag_W_hS +=  np.expand_dims(self.feedbackh, 1) *  self.Trace_W_hS
        #update tag hl
        self.Tag_W_hl +=  np.expand_dims(self.feedbackh, 1) *  self.Trace_W_hl 
        
        #feedback to latent
        dl = self.dtransfer(self.l_sens)
        self.fbh_transfer = self.dtransfer(self.fbh) #the feedback onto h is transferred through the hidden layer
        #W_hl_no_m = self.W_hl[:,:-3] #no match no bias, no time-part
        #W_hl_no_m = self.W_hl[:,10:30]
        
        W_hl_no_m = self.W_hl[:,obs_time().size:-(self.nblocks + 1)] #only learn back onto the stimulus input, not the time inputs (exclude the bias node)
        #inputs.obs_time().size
        
        self.W_hl_transpose = W_hl_no_m.T
        self.fhl = self.W_hl_transpose.dot(self.fbh_transfer) #for each node in l, all active h units are summed
        self.feedbackl = self.fhl * dl
        self.Tag_W_lx += np.expand_dims(self.feedbackl,1) * self.Trace_W_lx
        return

    def update_memory(self, zvec):
        # final z is 'do not gate'; nothing happens then
        if not zvec[-1] == 1:
            # else:
            gate_idx = np.argmax(zvec)
            l = self.l_in   #this excludes bias- and  match
            S_ = self.S.reshape( (self.nblocks, self.block_size) )
            W_ = self.W_Sl.reshape( S_.shape +  (l.size, ) ) 
            # project x->S (encode)
            Sproj = W_.dot(l)
            # store @ gated 'stripe'
            S_[gate_idx,:] = Sproj[gate_idx,:]
            # transform S back to its flat representation
            self.S = S_.reshape(self.S.shape)
        return
    


