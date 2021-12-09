#!/usr/bin/env python
"""
This file is part of the WorkMATe source code, provided as supplemental material
for the article:
    "Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers

Please see the README file for additional details.
Questions or comments regarding this file can be addressed to w.kruijne@vu.nl
-------------------
This file implements four functions for individual runs for the four tasks
tasks. The random seeds that each run get here yield 'illustrative'
convergence times.
"""
#%% Import modules
import numpy as np
import matplotlib.pyplot as plt

import world
import workmate_match
from m_runner import matchnet_regression, mean_std

#%% Time window constants

trial_print = 150 #printing of trials
max_time = 50000  #total number of training trials

#%% DMS task constants

n_switches = 5 # i.e. 6 stimulus sets, 3 stimuli per set
n_stim = 3
conv_crit = 0.85

#%% Run a trial
#np.random.seed(20) # a randnum [0- 1e10] > Use the np.random.seed NOT random.seed
#These are two different generators!
ncore = 4

def _run_trial(agent, env,log_q=False, **kwargs):
    """
    Generic function to run a trial:
    agent   : the agent being run
    env     : the environment implementing the task
    env     : if log_q=True, function returns both the reward and the output
                values in the q-layer

    Additional kwargs can be passed to be set in the environment (not used here)
    """
    if not kwargs is None:
        env.__dict__.update(kwargs)
    # init trial results:
    totrew = 0.0
    qlog = []
    match_per_trial = []
    tt_per_trial = []
    
    # step until obs == 'RESET', collect reward and q-values
    while True:
        #r, matches, tt_match = agent.step()
        r, matches, tt = agent.step()
        totrew += r
        if matches != None and tt != None:
            match_per_trial.append(matches)
            tt_per_trial.append(tt)
        if log_q:
            print((agent.x))
            qlog.append(agent.q)
        if agent.obs == 'RESET':
            break
        
    # format result:
    #print(match_per_trial)
    #print(tt_per_trial)
    retval = (totrew, np.array(qlog)) if log_q else totrew
    match_info = [match_per_trial, tt_per_trial]
    return retval, match_info

#%% Print information from trial to analyse the learning

def watch_trial(agent, env):
    """
    Function to watch a trial giving a specific agent and environment
    Returns the inputs, q-values, actions and rewards for each time step
    Returns the total reward received 

    """
    #This is a function that prints the agents strategy during a trial
    #This will give insight into its functioning
    
    #For each time step, I want to track
    xlog = []       #inputs
    qlog = []       #q-values
    zlog = []       #actions
    rlog = []       #rewards
    tlog = []
    envtlog = []
    obslog = []
    #trial_typelog = []
    totrew = 0      #total reward
    
    #step until obs == 'RESET', collect reward and q-values
    while True:
        
        totrew += agent.step()
        #print(agent.x)
        obslog.append(agent.obs)
        xlog.append(agent.x)
        qlog.append(agent.q)
        zlog.append(agent.z)
        rlog.append(agent.r)
        tlog.append(agent.t)
        envtlog.append(env.t)
        if agent.obs == 'RESET':
            trial_type = env.trial_type
            break
        
    retval = (totrew, obslog, trial_type, np.array(xlog), np.array(qlog), np.array(zlog), np.array(rlog), np.array(tlog), np.array(envtlog))
    return retval


#def run_dms(W_hx_pre, W_qh_pre, pre_stim, viz_converged=True, agent_type = 'W', print_all = 'off'):
def run_dms_match(matchnet, viz_converged=True, print_all = 'off', stim_type = 'alpha'):
    
    #stim_type can be 
    ###'alpha' > original small binary vectors
    ###'cifar' > learned latent vector representation CIFAR-10
    
    #np.random.seed(20)
    #n_switches = 5 # i.e. 6 stimulus sets, 3 stimuli per set
    
    #Initialise agent and task
    dms = world.DMS(n_stim=n_stim, n_switches=n_switches, stim_type = stim_type)
    agent = workmate_match.WorkMATe(dms, nblocks=2)
    agent.matchnet = matchnet
    print(agent)
          
    ### Initialize training:
    # buffer to store 'the moment of switching' (=convergence)
    iswi = np.zeros((n_switches+1)) 
    # buffer for last 500 trials:
    saved_p = np.zeros(1)
    total_buff = np.zeros(500) 
    # buffer for performance immediately after a switch
    swiperf = np.nan * np.ones((n_switches+1, total_buff.size))
    
    #Buffers for matchnet:
    x2_buff = np.zeros((agent.nblocks,1))
    tt_buff = np.zeros((agent.nblocks,1))
    performance_all_blocks = np.zeros(agent.nblocks)
    perf_match = [] #performance match_net
    
    # counters
    last=False
    i = 0 
    i_ = 0
    i_l = 0

    aa = True
    while aa:
        # run trial, get performance
        r, match_info = _run_trial(agent, dms)
        # increase i
        i += 1
        # was the trial correct?
        corr = ( r >= dms.bigreward )
        total_buff[0] = corr
        total_buff = np.roll(total_buff, 1)
        
        #buffers for match info
        x2_buff = np.append(x2_buff, np.array(match_info[0]).transpose(), axis=1)
        tt_buff = np.append(tt_buff, np.array(match_info[1]).transpose(), axis=1)        
        
        #if the past 100 trials were 85% correct, set is 'learned'
        if np.mean(total_buff[:100]) >= conv_crit and last == False:
            print('Convergence at {}\tSwitch to set {}'.format(i,dms.setnr+1))
            iswi[dms.setnr] = i
            # if criterion reached in less than 500 trials,
            # 'performanc post-switch' hasn't been logged yet -- do that now,
            # using only the trials with this set:
            if i < i_ + 500:
                swiperf[dms.setnr, :(i- i_)] = total_buff[:(i - i_)] # leaves nans for the rest of performance
            
            if dms.setnr == n_switches:
                dms.stimuli = dms.STIM
                last = True

            dms.switch_set()
            total_buff *= 0 # reset performance buffer
            i_ = i
            
        if last == True:
            i_l = i_l + 1
        
        if i_l >= 2000:
            break

        # @ iswi + 500: store post-switch performance:
        if i == i_ + 500:
            swiperf[dms.setnr, :] = total_buff

        # print progress:
        if i % trial_print == 0:
            #get matchnet performance
            for ff in range(agent.nblocks):
                performance = matchnet_regression(x2_buff[ff,:], tt_buff[ff,:])
                performance_all_blocks[ff] = performance.copy()
            perf_match.append(performance_all_blocks.copy())
            #reset buffers to zero
            x2_buff = np.zeros((agent.nblocks,1))
            tt_buff = np.zeros((agent.nblocks,1))
            
            #print performance in a list
            if print_all == 'on':
                print(i, '\t', np.mean(total_buff)) 
            saved_p = np.append(saved_p,np.mean(total_buff))
            
        if i >= max_time:
            aa = False
    print('Loop ended')
    
    return (dms, saved_p, iswi, agent, perf_match)

#%% Visualisation phase 2: Evaluation example

def plot_dms(saved_p,iswi):
    plt.rc('axes', labelsize=28)  
    plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=24)
    
    #print DMS performance in a graph
    arr = [i*trial_print for i in range(0,len(saved_p))]
    #get the markers on the line
    a = (np.floor(iswi/trial_print)).astype(int)    #find point closest below
    b = (np.ceil(iswi/trial_print)).astype(int)     #find point closest above
    
    iswi_p = np.zeros(len(iswi))
    for x in range(len(iswi)):                      #for all but the last one..
        if b[x] == len(saved_p):
            b[x] = b[x]                          #if its the last trial, set b[x] back to floor
        diff = saved_p[a[x] - 1] - saved_p[b[x] - 1]        #find difference in performance
        if diff < 0:                                #is b is bigger than a
           iswi_p[x] = b[x] - 1                          #choose b
        else:
           iswi_p[x] = a[x] - 1                         #if a is bigger choose a
    
    iswi_p = iswi_p.astype(int)
    
    fig, ax = plt.subplots(figsize=(10,8))
    #add switch positions to graph
    #iswi_index_insert = list(np.ceil(np.array(iswi)/trial_print) + range(n_switches+1))
    saved_p = list(saved_p)
    for i, value in enumerate(iswi):
        iswi_index_insert = np.ceil(value/trial_print) + i
        saved_p.insert(int(iswi_index_insert), conv_crit)
        arr.insert(int(iswi_index_insert), value)
    
    xposition = iswi
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--', linewidth=4)
        
    #plt.plot(iswi-trial_print,saved_p[iswi_p],'b*', markersize = 10)
    plt.plot(arr,saved_p,'r', linewidth=4)
    plt.xlabel('Trials')
    plt.ylabel('Ratio full task reward')
    plt.title('DMS task performance', fontsize=34)
    plt.grid()
    plt.show()
    return arr, saved_p

def plot_perf_DMS(iswi_buff):
    #Exclude non-converged agents
    iswi_conv = np.array([x for x in iswi_buff if np.all(x) == True])
    #How many excluded? 
    nr_conv = len(iswi_conv)
    #Not cumulative
    iswi_diff = np.c_[iswi_conv[:,0], np.diff(iswi_conv)]
    #Average + std for each level 
    iswi_avg, iswi_std = mean_std(iswi_diff)
    #Plot
    fig, ax = plt.subplots(figsize=(10,7))
    #ax.plot(np.arange(6)+1, iswi_avg, 'o')
    ax.set_axisbelow(True)
    plt.grid()
    plt.bar(np.arange(6)+1, iswi_avg/1000, color='navy', alpha=0.8)
    caplines = plt.errorbar(np.arange(6)+1, iswi_avg/1000, yerr=iswi_std/1000, ls='None', lolims=True, capsize=10, color='k')
    caplines[0].set_marker('_')
    
    plt.ylim([0,(iswi_avg[0]+iswi_std[0]+1000)/1000])
    plt.title('DMS trials until set convergence', fontsize=34)
    plt.ylabel('Trials (x1000)')
    plt.xlabel('DMS stimulus sets')
    return nr_conv

#%%
if __name__ == '__main__':
    run_dms_match(seed=1)
    