Learning to match Natural Scenes using Biologically Plausible Deep Reinforcement Learning
@author: Lieke Ceton

## Project information

This repository holds the official code and documentation of the project "Learning 
to match Natural Scenes using Biologically Plausible Deep Reinforcement Learning".

The project was jointly supervised by prof. dr. Pieter Roelfsema, head of the 
Vision & Cognition Lab (Netherlands Institute of Neuroscience, Amsterdam) and
prof. dr. Sander Bohté of the Machine Learning Lab (Centrum Wiskunde en 
Informatica, Amsterdam Science Park).

The one-year project was executed as part of the European Human Brain Project,
Task 3.7 Architectures and learning methods for hierarchical cognitive processing,
Objective 3.21 Flexible Cognitive Working Memory. An abstract was submitted
to Computational and Systems Neuroscience 2022 and a poster presentation on the 
project was held at the annual meeting of Organization The Graduate School 
Neurosciences Amsterdam Rotterdam in Woudschoten, Zeist. 

Questions or comments regarding this file can be addressed to p.roelfsema@nin.knaw.nl

## Project description

The project entails the extension of a Biologically Plausible Reinforcement Learning
Agent (WorkMATe[1]) to Deep Learning. This enables the learning of memory representations
of natural scenes (CIFAR-10 dataset) instead of short binary vectors. Secondly, research
was done into neural mechanisms underlying matching between sensory and memory inputs.
The results from an earlier neural comparator [2] were replicated and adapted to learn
within the RL agent. It was found that two sequential phases of learning are needed 
1) A motor babbling phase where unsupervised learning self-organises the matching networks
and 2) A reward driven phase that trains the agent to succesfully learn memory tasks
in an environment that needs matching. 

[1]"Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers
[2]"A self-organized neural comparator"
Ludueña, G. A., & Gros, C. (2013). 

## Dependencies

All files have been written and tested in Python 3.8, and make use of the modules:
- numpy
- scipy 
- matlibplot
- string
- random
- math
- sklearn 
- csv 

##Files in this repository

main.py 		     --> This file holds a short script to train a match network 
                         and visualise the performance
m_runner.py		     --> Functions to train phase 1: the match network
matchnet.py		     --> Architecture and learning rules match network 
workmate_match.py    --> Architecture and learning rules RL agent
world.py	         --> Environment DMS task RL agent 				  		
stimuli.py           --> Visual inputs used in DMS environment
tictoc.py		     --> Timing function
kaggle_feature.csv	 --> CIFAR-10 feature set extracted from Kaggle (Caleb Woy,	
https://www.kaggle.com/whatsthevariance/pytorch-cnn-cifar10-68-70-test-accuracy)

poster.pdf          --> Poster presented at ONWAR2021, Zeist
abstract.pdf        --> Abstract submission CoSyNe 2022
