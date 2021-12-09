# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:02:36 2021

This file holds the functions needed to train the matchnet during the motor 
babbling phase and visualise the results.

Functions m_runner:
    matchnet_regression 
    run_match 
    plot_eval
    plot_perf_match

@author: Lieke Ceton
"""
#%%Import modules
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib.pyplot as plt
import math

#%% Time window constants
wiw = 150               #Window of trials: Regression done every 250 trials
w_eval = 500            #Number of trials used for evaluation
max_time = 15000         #max number of trials run

#%% Run phase 1: Motor babbling

def mean_std(buff):
    m_avg = np.mean(buff,axis=0)
    m_std = np.std(buff,axis=0)
    return m_avg, m_std

def matchnet_regression(m_out, trial_type):
    """Evaluation method: Perform logistic regression on the matchnet output 
    (m_out) with binary class labels (trial_type) and return score [0-1]"""
    model = LogisticRegression(solver='liblinear', random_state=0) #init model
    x = m_out.reshape(-1,1)            #select match values of first window
    y = trial_type                      #select trial types of first window
    model.fit(x, y)                     #fit the model
    performance = model.score(x, y)     #compare true class labels to model prediction
    return performance

def run_match(agent, dms):
    """Training phase 1: Motor babbling. The agent trains (agent.step_match()) 
    as long as the performance is not above threshold. When max_time is reached,
    the training is also aborted. The trained agent, the index of convergence
    and performance at each wiw (window of trials) is returned."""
    
    #Init environment (for number of actions and length of stimulus type)
    dms = dms
    #Init WorkMATe agent
    agent = agent
    #For each memory block, a matchnet is also intialised
    print('Task and agent initialised')
    
    #Initialise training
    i = 1                                           #counter
    i_conv = None                                   #index of convergence                     
    
    #Create buffer for match values, trial types and cosine correlation
    val_buff = np.zeros((3, agent.nblocks, w_eval))
    performance_all_blocks = np.zeros(agent.nblocks)#performance both matchnets
    perf_match = []                                 #performance matchnet

    #Train the matchnet layers
    aa = True
    while aa:
        #Do one matching trial, get match value, 
        m_out, trial_type, cos_corr = agent.step_match()
        
        #Add to buffer
        new_val = np.array([m_out, trial_type, cos_corr])
        val_buff[:,:,0] = new_val
        
        #Get performance after a window of trials has gone
        if i%wiw == 0:
            #Get logistic regression score for each matchnet independently
            for ff in range(agent.nblocks):
                #Get the matchnet performance through regression of m_out and tt_out
                performance = matchnet_regression(val_buff[0,ff,0:wiw], val_buff[1,ff,0:wiw])
                performance_all_blocks[ff] = performance.copy()
            perf_match.append(performance_all_blocks.copy())
            #Note index of convergence when performance of all matchnets is 
            #99% for the first time
            if (performance_all_blocks >= 0.99).sum() == agent.nblocks and i_conv == None: 
                i_conv = i
        #Stop training when trials exceed max_time
        if i >= max_time: 
            aa = False 
        
        val_buff = np.roll(val_buff, 1, axis=2)
        i += 1 #Add to counter

    return agent, val_buff, i_conv, perf_match

#%% Visualisation of phase 1

#Font sizes
plt.rc('axes', labelsize=24)  
plt.rc('xtick', labelsize=20)    
plt.rc('ytick', labelsize=20)

def plot_eval(val_buff, m=0):
    """"Visualisation of the match value clustering at the end of training, the
    evaluation trials are the last trials of training."""
    #Graph set-up
    fig, ax = plt.subplots(figsize=(10,7))    
    plt.title('Match value evaluation (m_block={})'.format(m), fontsize=30)
    plt.xlabel('Ordered evaluation trials (non-match --> match)')
    plt.ylabel('Match value')
    trials = range(len(val_buff[0,m,:]))
    ax.set_xlim([0, len(val_buff[0,m,:])])
    
    #Order values based on cosine similarity
    i_sort_coscor = np.argsort(val_buff[2,m,:])
    val_buf_sort = val_buff[:,:,i_sort_coscor]
    
    #Plot the match outputs of the network
    plt.plot(trials, val_buf_sort[0,m,:], 'o', color='navy', markersize = 9, label='Instance')
    #Get the binned average of the match outputs
    bin_means_2, bin_edges, binnumber = stats.binned_statistic(range(w_eval),
                val_buf_sort[0,m,:], statistic='median', bins=np.array(range(100))*20+20)
    #Plot the average match value
    plt.plot(np.array(range(99))*20+20, bin_means_2, 'r', zorder = 10, label='Average', linewidth = 6)
    #Plot cosine similarity of the inputs
    plt.plot(trials, val_buf_sort[2,m,:], 'g', label = 'Similarity ', linewidth = 6) 
    #Plot the change of trial type based on the cosine similarity going to 1
    border = np.where(np.around(val_buf_sort[2,m,:], 5)==1)[0][0]
    ax.axvspan(border, w_eval, facecolor='grey', alpha=0.5)
    
    plt.legend(loc='best', fontsize=22)
    plt.show()    
    return val_buf_sort

def plot_perf_match(perf_match_buff, i_conv_buff):
    """Visualisation of the performance of the matching network in time. A regression 
    score of 0.5 indicates random, 1 is perfect classification"""
    #Graph set-up
    fig, ax = plt.subplots(figsize=(10,7))    
    plt.title('Matchnet performance \n Logistic regression classification', fontsize=26)
    plt.xlabel('Trials')
    plt.ylabel('Ratio correctly predicted')
    
    nr_match = np.array(perf_match_buff).shape[1]
    nr_mem = np.array(perf_match_buff).shape[2]
    
    x_array = np.arange(nr_match)*wiw+wiw 
    for i in range(nr_mem):
        perf = np.array(perf_match_buff)[:,:,i]
        m_avg, m_std = mean_std(perf)
        #Plot the match value performance using the regression scores
        plt.plot(x_array, m_avg, '-o', linewidth=5, markersize=12, label="m={}".format(i))
        #Plot the standard deviation of the match values 
        ax.fill_between(x_array, m_avg-m_std, m_avg+m_std,alpha=0.3)
    
    #Plot the index of convergence (mean and standard deviation)
    i_avg, i_std = mean_std(i_conv_buff)
    plt.axvline(i_avg, c='k', linewidth=5, label='i_conv')
    ax.axvspan(i_avg-i_std, i_avg+i_std, alpha=0.2, color='black')
    
    if math.isnan(i_avg):
        plt.xlim(0, max_time)
        print('Convergence >99% is not (yet) reached for all agents')
    else:
        plt.xlim(0, i_avg+1000) #plot until a bit after i_conv
    
    plt.grid()
    plt.legend(loc='best', fontsize=22)
    plt.show()

