# Learning to match Natural Scenes using Biologically Plausible Deep Reinforcement Learning

## Project information

This repository holds the official code and documentation of the project "Learning 
to match Natural Scenes using Biologically Plausible Deep Reinforcement Learning".

The project was supervised by prof. dr. Pieter Roelfsema, head of the 
Vision & Cognition Lab (Netherlands Institute of Neuroscience, Amsterdam) and
prof. dr. Sander Bohté of the Machine Learning Lab (Centrum Wiskunde en 
Informatica, Amsterdam Science Park).

The project was executed as part of the European Human Brain Project,
Task 3.7 Architectures and learning methods for hierarchical cognitive processing,
Objective 3.21 Flexible Cognitive Working Memory. An abstract was submitted
to Computational and Systems Neuroscience 2022 and a poster presentation on the 
project was held at the annual meeting of Organization The Graduate School 
Neurosciences Amsterdam Rotterdam in Woudschoten, Zeist. 

## Project description

The project entails the extension of a Biologically Plausible Reinforcement Learning
Agent (WorkMATe[1]) to Deep Learning. This enables the learning of memory representations
of natural scenes (CIFAR-10 dataset) instead of short binary vectors. Secondly, research
was done into neural mechanisms underlying matching between sensory and memory inputs.
Results from an earlier neural comparator [2] were replicated and adapted to learn
within the RL agent. It was found that two sequential phases of learning are needed: 
1) A motor babbling phase where unsupervised learning self-organises the matching networks
2) A reward driven phase that trains the agent to succesfully learn memory tasks
in an environment that needs matching. 

[1]"Flexible Working Memory through Selective Gating and Attentional Tagging"
Wouter Kruijne, Sander M Bohte, Pieter R Roelfsema, Christian N L Olivers
<br> [2]"A self-organized neural comparator"
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

## Files in this repository

- main.py &rarr; This file holds a short script to train a match network and visualise the performance
- m_runner.py &rarr; Functions to train phase 1: the match networ
- matchnet.py &rarr; Architecture and learning rules match network
- w_runner.py &arr; Functions to train phase 2: the RL agent 
- workmate_match.py &rarr; Architecture and learning rules RL agent
- world.py &rarr; Environment DMS task RL agent 				  		
- stimuli.py &rarr; Visual inputs used in DMS environment
- tictoc.py &rarr; Timing function
- kaggle_feature.csv &rarr; CIFAR-10 feature set extracted from Kaggle (Caleb Woy,	
https://www.kaggle.com/whatsthevariance/pytorch-cnn-cifar10-68-70-test-accuracy)
- poster.pdf &rarr; Poster presented at ONWAR2021, Zeist
- abstract.pdf &rarr; Abstract submission CoSyNe 2022

## File structure

![image](https://user-images.githubusercontent.com/71390417/145237082-9c3a523e-addf-4a85-b175-073e38b2f991.png)
