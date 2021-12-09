# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:21:44 2021

This file holds the stimuli that are used in the world to represent cues. 

obs_time        --> Stimulus representing time
match_cifar     --> Natural scenes for phase 1 learning 
obs_cifar       --> Natural scenes for phase 2 learning 
match_alpha     --> Alphabetic letters for phase 1 learning 
obs_alpha       --> Alphabetic letters for phase 2 learning 

Detailed information on the stimuli can be found in README.txt
@author: Lieke Ceton
"""

#%% Dependencies
import numpy as np 
import string
from random import sample
import csv
from sklearn.preprocessing import normalize

#%% Time cell coding

maxtime  = 10
# Time vectors are created by convolving a response vector 
# with an identity matrix, yielding [maxtime] rows of time cell responses,
# each peaking at a unique, consecutive time.
z = [0.1, 0.25, 0.5, 1, 0.5, 0.25, 0.1]
crop = int((len(z)-1)/2) # the '3'-cropping here removes edge artefacts from convolution; 
# Time cell 0 (at row 0) peaks at the first moment in time (column 0).
tmat = np.vstack([np.convolve(z, t)[crop:maxtime + crop] for t in np.eye(maxtime)])

def obs_time(t=0):
    """Vector that represents time"""
    return tmat[t]

#%% CIFAR-10 observations for both learning phases

#CIFAR-10 features are extracted from a pre-trained CNN (Caley Woy, see README)
#They are the activity vectors of the second fully connected layer.
#load .csv file
with open("CIFAR_10_kaggle_feature_2.csv", 'r') as f:
    csv_features = list(csv.reader(f, delimiter=","))

all_feat = np.array(csv_features[1:], dtype=np.float)   #get the first row out
match_dict = normalize(all_feat[:,1:-2])                #normalize
feat_sample = all_feat[0:500,1:-2]                      #Sample the first 500 features/images
cifar_dict = normalize(feat_sample)                     #normalise 
        
def match_cifar():
    """Stimuli for phase 1 learning, random natural scenes from CIFAR-10 dataset"""
    a = np.random.choice(match_dict.shape[1])
    return match_dict[a]

def obs_cifar(obs=1):
    """Stimuli for phase 2 learning, a specific set of CIFAR-10 stimuli is selected"""
    return cifar_dict[obs]

#%% Alpha observations for both learning phases

#Construct stimulus dictionary
stimbits = 10 #length of stimuli
#Construct binary stim_repres
binstr = '0{}b'.format(stimbits)
binstrings = [format(i, binstr) for i in range(2**stimbits)]
tobinarr = lambda s : np.array([float(c) for c in s])
Dx = np.vstack([tobinarr(i) for i in binstrings]) #--> a 

shuffle = sample(range(len(Dx)),len(Dx)) #shuffle the rows randomly 
Dx = Dx[shuffle,:] 

# Dx now is a matrix of 128 x 7 bits. 'stimbits' is a dict that will order the 
# first 52 of these in a lookup table, #why not choose 2**6 when you only use the first 52? (LJC)
chars = string.ascii_lowercase + string.ascii_uppercase
stimdict = dict(list(zip( chars, Dx )))

# Stimuli with these 5 letters are used in prosaccade/antisaccade, and here made
# linearly separable, cf. Rombouts et al., 2015
stimdict['g'] = np.zeros(stimbits)
stimdict['p'] = np.eye(stimbits)[0]
stimdict['a'] = np.eye(stimbits)[1]
stimdict['l'] = np.eye(stimbits)[2]
stimdict['r'] = np.eye(stimbits)[3] #why? this ruins the neat dictionary that you just made.. (LJC)

# digits, used in 12-AX, are added to the stimdict in a similar manner
digdict = dict( 
    [(d,Dx[i + 2**(stimbits-1) ]) for i,d in enumerate(string.digits) ])
stimdict.update(digdict)

len_Dx = Dx.shape[0]

def match_alpha():
    """Stimuli for phase 1 learning, random vector selected from binary stimuli"""
    rand_int = np.random.choice(len_Dx)
    return Dx[rand_int,:]

def obs_alpha(obs='A'):
    """Stimuli for phase 2 learning, all lower and uppercase letters (52 stimuli)"""
    # return the row of activity from the selected stimdict index as the observation
    return stimdict[obs]


