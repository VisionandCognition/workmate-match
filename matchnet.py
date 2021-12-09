# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:27:33 2021

This file is Python code based on Ludueña, G. A., & Gros, C. (2013). 
A self-organized neural comparator. Neural Computation, 25(4), 1006–1028. 
https://doi.org/10.1162/NECO_a_00424
-------------------
The MatchNet class defined in this file implements a one-layer version of the neural comparator:
Hebbian learning is performed when training MatchNet on MatchNet_Data
At each time point:
1-  Two vectors are sampled from MatchNet_Data concatenated into one x1 activity vector
2-  A feedforward pass computes the match value x2_out (max(x2))
3-  A local anti-Hebbian learning rule updates the weights of each neuron
I have adapted to a one-layer simplied version, but extra layers can easily be added.

The option of having an injective transformation between the two input vectors is also available.
After training, the match values of the evaluation data (default 10%) are plotted
Performance is currently not quantified, but match and non-match trials can be 
split on a 1D-scale with low error.

@author: Lieke Ceton
This code is my own and was not written in collaboration with Ludueña and Gros.
Questions or comments regarding this file can be addressed to lieke.c@nin.knaw.nl

"""

#%% Dependencies
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tictoc import tic, toc

#%% Functions for matchnet architecture

def drop_out(weights, p_conn):
    """Drop-out in weight matrices, mask is saved to perform same drop-out again"""
    N = weights.size
    #Create a binary matrix with the same size as the weights, p_conn indicates how many non-zero connections
    mask = np.random.choice([0, 1], size=(N,), p=[1-p_conn, p_conn]).reshape((weights.shape[0], weights.shape[1]))
    #Overlay the weights with the mask 
    new_weights = weights*mask
    return new_weights, mask

def scale_linear_input(inp_trans, min_range, max_range):
    """Scale a vector input between a min and max range"""
    inp_std = (inp_trans - min(inp_trans)) / (max(inp_trans) - min(inp_trans))
    inp_scaled = inp_std*(max_range - min_range) + min_range
    return inp_scaled

def cosine_similarity(inp1, inp2):
    """"Cosine similarity between two input vectors inp1 and inp2. 
    Used as an extra evaluation tool when judging the comparator performance"""
    vec_norm = np.linalg.norm(inp1)*np.linalg.norm(inp2) #calculate the norm
    if vec_norm == 0: #if the length of the vector is 0, the cosine correlation is also zero
        cos_corr = 0
    else:
        cos_corr = np.dot(inp1, inp2)/vec_norm #definition cosine similarity
    return cos_corr

#%% 
class MatchNet_Data(object):
    """This class defines the data that the matchnet is trained on. 
    During each training step, two inputs are sampled from the matchnet_data."""      
    
    def __init__(self, N=30, encoding='identity', seed = 30):
        super().__init__()
        #np.random.seed(seed) #seed can be used for reproducibility
        
        #Matchnet data type and encoding relationship between matching inputs
        self.N = N                  #input length
        self.encoding = encoding    #encoding can be identity (default) and linear
        #identity --> match vectors are the exact same, linear --> a fixed transformation between them
        self.p_eq = 0.5             #probability of match (ratio trial types)
        
        #Define fixed matrix transformation for linear encoding
        if encoding == 'linear':
            self.A = np.random.uniform(-1, 1, (self.N, self.N))
        else: self.A = 0   
        
    def sample_input(self):
        """Sample two inputs from the matchnet_data"""
        #Get a trial type --> 0 is no-match, 1 is match
        trial_type = np.random.choice([0,1], p=[1-self.p_eq, self.p_eq]) #p_eq is match probability
        
        aa = True
        while aa:
            #Create two inputs: y (inp1) and z(inp2)
            self.inp1, self.inp2 = np.random.sample((self.N,)), np.random.sample((self.N,))

            if trial_type == 1: #match
                if self.encoding == 'identity':
                    self.inp2 = np.copy(self.inp1) #match inputs are the exact same
                if self.encoding == 'linear':
                    inp_transform = self.A.dot(self.inp1) #dot product with fixed matrix
                    self.inp2 = scale_linear_input(inp_transform, -1, 1) #normalized to fit between 1 and -1
        
            #Calculate cosine similarity between incoming vectors for later evaluation
            cosine_corr = cosine_similarity(self.inp1, self.inp2)
            if cosine_corr != 0: #if by accident one of the vectors is zero, try again
                aa = False
        
        return self.inp1, self.inp2, trial_type, cosine_corr  

#%%
class MatchNet(object):
    """The matchnet is the network that learns to output different values for 
    match and non-match trials. This class holds the architecture and the learning"""
    def __init__(self, inp1, inp2, seed = 30):
        #np.random.seed(seed)
        self.lr = 0.002             #original lr = 0.003
        self.alpha_low = 3.5        #sigmoid slope for N < 400 !! this factor has a lot of influence
        self.alpha_high = 1.0       #sigmoid slope for N >= 400
        self.p_conn = 1             #probability of connectivity between layers of the network
        self.phase = 1              #Phase of learning
                 
                   #1 is fully connected
        #Define input size N
        if len(inp1) != len(inp2):
            raise Exception("The length of the two inputs is not the same")
        self.N = len(inp1)
        
        self.nx1 = 2*self.N              #size layer 1
        self.nx2 = round((self.N)/2)     #size layer 2 (max is outcome)
        
        def init_layer(pre_layer_size, post_layer_size, p_conn):
            """Initialise one layer of the network"""
            w = np.random.sample((post_layer_size, pre_layer_size))*0.2-0.1
            weights, mask = drop_out(w, p_conn)
            delta_weights = np.zeros_like(weights)
            layer = np.zeros(post_layer_size)
            return weights, mask, delta_weights, layer

        #A feedforward network of one layer, save all weights for analysis
        self.W_x1x2, self.mask_W_x1x2, self.delta_W_x1x2, self.x2 = init_layer(self.nx1, self.nx2, self.p_conn) #original p_conn = 0.8
        return
        
    def step(self, inp1, inp2):
        """During each trial, the matchnet takes a step: feedforward + feedback"""
        #The input is the concatenated activity (length 2N)
        x1 = 0
        x1 = np.concatenate((inp1, inp2))
        #Do a feedforward pass and compute the output by taking the max
        x2 = self.feedforward(x1)
        x2_out = max(x2)
        
        #Learning stabilises in the second reward-driven phases
        if self.phase == 2:
            return x2_out, x2
        
        #Locally learn the weights using the Anti-hebbian learning rule
        self.feedback(x1, x2)
        return x2_out, x2
    
    def feedforward(self, x1):
        """Feedfoward processing produces a scalar output activation --> the match value"""
        #Transfer function is the hyperbolic tangent, information is passed forward by a dot product
        x2_in = self.W_x1x2.dot(x1)
        x2 = np.tanh(self.alpha_low*x2_in)
        #A threshold can be used for binary classification > not needed here for us, we want a match scalar
        #The match value is the most active second layer neuron
        return x2
    
    def feedback(self, x1, x2):
        """Feedback processing learns the weights following unsupervised anti-Hebbian learning"""
        def learn_weights(W, mask, a1, a2):
            #Anti-Hebbian Learning rule
            delta_W = -self.lr*np.outer(a2, a1) #a learning factor for each combination of the pre and postsynaptic neurons
            #Don't forget that not each connection exists >> there is a p_conn below 1! Apply the mask!
            delta_W = delta_W*mask
            W += delta_W
            
            #The weights are normalised to the sum of all connections to one postsynaptic neuron 
            #This is a local homeostatic factor
            for item in range(W.shape[0]):
                norm_factor = np.sqrt(np.sum(W[item]*W[item]))
                W[item] = W[item]/norm_factor
            return W, delta_W
    
        #Update the weights by delta using the presynaptic and postsynaptic activity
        self.W_x1x2, self.delta_W_x1x2 = learn_weights(self.W_x1x2, self.mask_W_x1x2, x1, x2)
        
#%% Multiple learning steps
def train_matchnet(matchnet, matchnet_data, t_max=10000, t_eval=1000):
    """Training the matchnet: For each t in t_max, sample inputs and take a step"""
    eval_data = []
    for i in range(t_max):
        #There is no feedback, no sequentiality, each trial is one independent time step
        #Sample two inputs from the matchnet data
        inp1, inp2, trial_type, cosine_corr = matchnet_data.sample_input()
        #Train one step
        x2_out, x2 = matchnet.step(inp1, inp2)
        
        #print trial_progress
        if i % 1000 == 0:
            print(i)
        
        #Save evaluation data
        #Save the t_eval results (10%) in paper in a matrix called eval_data
        if i >= (t_max-t_eval):
            data_row = [x2_out, x2, trial_type, cosine_corr] #save the input vectors, the trial type and the output
            eval_data.append(data_row)
    
    #Detangle the list columns in rows
    eval_dt = list(map(list, zip(*eval_data))) #in order: x2 output, x2, trial_type, cosine_corr   
    return matchnet, eval_dt #norm_data

#%% Run the neural comparator with random or sparse CIFAR-10 data
def run_ncomp(t_max, t_eval, N, encoding):
    """Complete run of the matchnet training: based on N and encoding, the matchnet_data
    and matchnet are initialised. After, it is trained for t_max timesteps"""
    #Initialize the data (data type and encoding)
    matchnet_data = MatchNet_Data(N, encoding=encoding) 
    #Initialize the network using one sample of the data
    inp1, inp2, trial_type, cosine_corr = matchnet_data.sample_input()
    matchnet = MatchNet(inp1, inp2) 

    tic()
    #Train the network in an unsupervised manner, print every dividable by 10000
    matchnet, eval_dt = train_matchnet(matchnet, matchnet_data, t_max, t_eval) 
    toc() #Print elapsed time
    return matchnet, matchnet_data, eval_dt

#%% Visualise
#Font sizes
plt.rc('axes', labelsize=24)  
plt.rc('xtick', labelsize=20)    
plt.rc('ytick', labelsize=20)

def plot_ncomp_eval(eval_dt):
    """"Visualisation of the match value clustering at the end of training, the
    evaluation trials are the last trials of training."""
    #Graph set-up
    fig, ax = plt.subplots(figsize=(10,7))    
    plt.title('Match value evaluation', fontsize=30)
    plt.xlabel('Ordered evaluation trials (non-match --> match)')
    plt.ylabel('Match value')
    trials = range(len(eval_dt[0]))
    ax.set_xlim([0, len(eval_dt[0])])
    
    #Order values based on cosine similarity
    i_sort_coscor = np.argsort(eval_dt[3])    
    #For now, do not use the first row
    eval_dts = np.array([eval_dt[0], eval_dt[2], eval_dt[3]])[:,i_sort_coscor]
    
    #Plot the match outputs of the network
    plt.plot(trials, eval_dts[0], 'o', color='navy', markersize = 9, label='Instance')
    #Get the binned average of the match outputs
    bin_means_2, bin_edges, binnumber = stats.binned_statistic(range(len(eval_dt[0])),
                eval_dts[0], statistic='median', bins=np.array(range(100))*20+20)
    #Plot the average match value
    plt.plot(np.array(range(99))*20+20, bin_means_2, 'r', zorder = 10, label='Average', linewidth = 6)
    #Plot cosine similarity of the inputs
    plt.plot(trials, eval_dts[2], 'g', label = 'Similarity ', linewidth = 6) 
    #Plot the change of trial type based on the cosine similarity going to 1
    border = np.where(np.around(eval_dts[1], 10)==1)[0][0]
    if border == 0:
        border = np.where(np.around(eval_dts[1], 10)==0)[0][0]
    ax.axvspan(border, len(eval_dt[0]), facecolor='grey', alpha=0.5)
    
    plt.legend(loc='best', fontsize=22)
    plt.show()    
    return eval_dt

#%% Example run
if __name__ == '__main__':
    
    t_max = 12000         #number of training steps (10**7 in the paper)
    t_eval = 1000         #number of steps that are evaluated (10**6 in paper)
    N = 30                #length of the input vector
    
    #Train the matchnet
    matchnet, matchnet_data, eval_dt = run_ncomp(t_max, t_eval, N, 'identity')
    #Plot the evaluation data
    plot_ncomp_eval(eval_dt)

    #Run the trained neural comparator for new samples (choose, inp4, inp5, inp6)
    #High values for non-match trials, low values for match trials
    inp4, inp5 = np.random.sample((N,)), np.random.sample((N,))
    inp6 = inp4    
    x4 = matchnet.step(inp4, inp6)
    print(x4[0])