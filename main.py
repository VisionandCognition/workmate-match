# -*- coding: utf-8 -*-
""" Deep WorkMATe with matchnet implementation
Created on Tue Nov 30 11:22:34 2021

Main file that holds 
1) The learning of a neural comparator in isolation 
2) The learning of the comparator within a Deep Reinforcement Learning agent
This is phase 1 in a 2-phase learning paradigm (Details in README)

For simplicity, the second learning phase (Reward-driven learning), is 
not included here. 

@author: Lieke Ceton
"""

#%% Import modules
import numpy as np

import workmate_match
import world
from matchnet import run_ncomp, plot_ncomp_eval
from m_runner import run_match, plot_eval, plot_perf_match
from w_runner import run_dms_match, plot_dms, plot_perf_DMS

#%% Replication results Neural Comparator 

#This is a one-layer simplification of the original comparator module
#Below the comparator is trained for a given t_max steps,
#inputting random vectors with length N, which are the exact same in match trials
#(identity encoding, default) or are linearly transformed (linear encoding)

t_max = 10000             #number of training steps (10**7 in the paper)
t_eval = 1000             #number of steps that are evaluated (10**6 in paper)
N = 30                    #length of the input vector
encoding = 'identity'     #can be identity or linear

#Train the matchnet
matchnet, matchnet_data, eval_dt = run_ncomp(t_max, t_eval, N, encoding)

#Plot the evaluation data
plot_ncomp_eval(eval_dt)

#%% Phase 1: Motor babbling

#This section implements the above neural comparator in a Deep Reinforcement
#Learning agent and trains the match network in an embedded fashion. Each memory
#block has its own independent match network.
#The inputs are now transformed CIFAR-10 images and encoding is always identity
#Evaluation is extended to performance in time based on the prediction of 
#a logistic regression model, this gives a score for how easy match and non-match
#trials can be separated and gives the cue for the end of training.

n = 5               #Number of agents trained
n_mem = 2           #Number of memory blocks+matching networks  per agent
n_eval = 2          #Agent selected for evaluation example
stimulus = 'cifar'  #Stimulus type can be cifar or alpha

#Initialise DMS environment --> crucial for embedding the match network
dms = world.DMS(n_stim=3, n_switches=5, stim_type = stimulus)

#Initialise buffers
i_conv_buff = np.zeros(n)   #indices of convergence
perf_match_buff = []        #matching performance for blocks of trials

for i in range(n):
    #Initialise agent 
    agent = workmate_match.WorkMATe(dms, nblocks=n_mem)  
    #Train the matching network
    agent, val_buff, i_conv_buff[i], perf_match = run_match(agent, dms) 
    #Add the performance to the buffer
    perf_match_buff.append(perf_match)
    #Return the trained agent, evaluation buffer, index of convergence, performance
    
    if i == n_eval: #Save the value buffer of the selected agent
        n_eval_data = val_buff.copy() 

#EVALUATE TRAINING: Logistic regression performance in time
#i_conv is the index of convergence (performance all n > 99%)
plot_perf_match(perf_match_buff, i_conv_buff) 

#EVALUATION EXAMPLE: Plot batch of match value outcomes after training
#m indicates the selected memory block
#A cosine similarity of 1 indicates match trials
plot_eval(val_buff, m=0) 

#%% Phase 2: Reward-driven Learning

#Run phase 1 for one agent --> get matchnets to use for each agent
#This can be done as the matchnets are shown to converge to very similar networks
agent = workmate_match.WorkMATe(dms, nblocks=n_mem)  
agent, val_buff, i_conv_buff[i], perf_match = run_match(agent, dms) 
matchnet = agent.matchnet
#Turn off learning matching network
for item in matchnet:
    item.phase = 2  #when initialised, this is set to 1
 
n = 5               #Number of agents trained
pr = 'off'          #Printing settings: on or off
st = 'cifar'        #Stimulus type: cifar (default) or alpha
n_eval = 5          #Agent selected for evaluation example 

iswi_buff = []      #Initialise buffer
example_buff = []   #Save for the selected example agent

for i in range(n):
    #Run agents and save convergence per stimulus in buffer
    dms, saved_p, iswi, agent, perf_match_dms = run_dms_match(matchnet, print_all = pr, stim_type = st)
    iswi_buff.append(iswi)
    
    if n_eval == i: #Save specific data of the selected agent
        example_buff = [dms, saved_p, agent, perf_match_dms]

#EVALUATE TRAINING: Bar plot of trials needed to learn a level
nr_conv = plot_perf_DMS(iswi_buff)
#Only the converged agents are plotted
print('{} of {} agents have converged on the task'.format(nr_conv, n))

#EVALUATION EXAMPLE: Performance over trials
plot_dms(saved_p,iswi)
#Each dashed line represents a switch of the stimulus set used for training
#Meta-learning is observed: Learning is faster after the task is learned on another set


