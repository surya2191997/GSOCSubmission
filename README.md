# GSOCSubmission

This repository contains code for GSoC 2019 selection tasks for TMVA-CERN 

##  Project 3: Development of LSTM and GRU layers in TMVA.


### Task 1 and 2: RNN_TMVA

Code for an RNN classifier implemented using TMVA. The dataset has been taken from TMVA classification tutorial. The task is to classify any event (say in the Large Hadron Collider) as signal or background.

Directions for running: 

* .x ./RNNClassification.C in root command-line.



### Task 3 and 4: LSTM

Code for forward and backward pass for a single LSTM cell. 

* LSTM.cpp     :   Contains functions for forward and backward pass for a single LSTM cell
* operations.h :   Contains functions for matrix and vector operations 

Directions for running:

* g++ LSTM.cpp
* ./a.out
