#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:57:59 2019

@author: carsault
"""
import os, errno
import pickle
import torch
from utilities import chordUtil
from utilities.chordUtil import *
from utilities import testFunc
from utilities.testFunc import *
from utilities import distance
from utilities.distance import *
#from ACE_Analyzer import ACEAnalyzer
#from ACE_Analyzer.ACEAnalyzer import *
import numpy as np
import time
#%% define grid search
foldName = "modelSave200908BadLR"
serv = "cedar"
DaysOfTraning = "00"
HoursOfTraining = "3"
decimList = [1]
modelType = ["mlpDecim"]
#batch = 64
batch = 500

#decimList = [1]
#modelType = ["lstmDecim"]
#batch = 50
layer = [1,2,3]
bottleN = [10,20,50,100]
hidden = [100,200,500,1000]
dropOut = [0, 0.2, 0.4, 0.6, 0.8]


randm = [1]
alpha = ['a0']
#alpha = ['functional']

dictModel = {}
scorModel = np.zeros((len(modelType), len(randm), len(alpha), len(layer), len(bottleN), len(hidden), len(dropOut)))
paramModel = np.zeros((len(modelType), len(randm), len(alpha), len(layer), len(bottleN), len(hidden), len(dropOut)))
# open sum up file
s = 0




sumUp = open("results.txt", "w")
for model in modelType:
    #for decim in decimList:
    for rand in randm:
        for alph in alpha:
            l_it=0
            for l in layer:
                bn_it=0
                for bn in bottleN:
                    hd_it=0
                    for hd in hidden:
                        dp_it=0
                        for dp in dropOut:
                            dataFolder = alph + "_1_" + str(rand)
                            str1 = ''.join(str(e) for e in decimList)
                            modelName = dataFolder + "_" + str1 + "_" + model + str(s)
                            res = pickle.load(open(foldName + '/' + modelName + '/' + "res" + modelName + ".pkl", "rb" ) )
                            sumUp.write(str(s) + " , " + str(res["bestAccurTest"]) + " --layer " + str(l) + " --latent " + str(bn) + " --hidden " + str(hd) + " --dropRatio " + str(dp) + "\n")
                            scorModel[0][0][0][l_it][bn_it][hd_it][dp_it] = res["bestAccurTest"]
                            paramModel[0][0][0][l_it][bn_it][hd_it][dp_it] = res['numberOfModelParams']
                            s += 1
                            dp_it +=1
                        hd_it+=1
                    bn_it+=1
                l_it+=1            #sumUp.write(h+lat+l+mT)
           
sumUp.close()
dictModel['scor'] = scorModel
dictModel['param'] = paramModel
#launch.close()
sauv = open(foldName + "_gridSearchMLP.pkl","wb")
pickle.dump(dictModel,sauv)
sauv.close()              
print("analyses completed")
