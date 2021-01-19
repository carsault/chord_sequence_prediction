#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:12:57 2020

@author: carsault
"""

#%%
from os import listdir
from os.path import isfile, join
import pickle

from utilities import chordUtil
from utilities.chordUtil import *
from ACE_Analyzer import ChordsToChromaVectors
from ACE_Analyzer.ChordsToChromaVectors import *

#Function to left rotate arr[] of size n by d*/ 
def leftRotate(arr, d, n): 
    for i in range(d): 
        leftRotatebyOne(arr, n) 
  
#Function to left Rotate arr[] of size n by 1*/  
def leftRotatebyOne(arr, n): 
    temp = arr[0] 
    for i in range(n-1): 
        arr[i] = arr[i+1] 
    arr[n-1] = temp 

alpha = "a0"
dictChord, listChord = chordUtil.getDictChord(eval(alpha))

mypath = "histo_output"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#%%
listModel = ["mlpDecim","mlpDecimFamily","mlpDecimKey","mlpDecimBeat","mlpDecimKeyBeat","mlpDecimAug"]
for model in listModel:
    hist = pickle.load( open( mypath + "/" + model + "_" + alpha + "_1histoChord.pkl", "rb" ) )
    
    cor_listMaj = [0] * (len(dictChord)-1)
    cor_listmin = [0] * (len(dictChord)-1)
    for key,value in hist.items():
        try:
            root, qual = key.split(":")
            for chord in range(len(dictChord)-1):
                root2, qual = listChord[chord].split(":")
                delRoot = delta_root(root,root2)%12
                if qual == "maj":
                    cor_listmin[delRoot*2] += hist[key][chord]
                else:
                    cor_listmin[delRoot*2+1] += hist[key][chord]
    
        except:
            if key == "N":
                root = key
                continue
            else:
                root = key
            for chord in range(len(dictChord)-1):
                root2, qual = listChord[chord].split(":")
                delRoot = delta_root(root,root2)%12
                if qual == "maj":
                    cor_listMaj[delRoot*2] += hist[key][chord]
                else:
                    cor_listMaj[delRoot*2+1] += hist[key][chord]
    print(cor_listMaj)
    #with open(mypath + "/" + model + "_" + alpha + "cor.pkl", "wb" ) as fp:   #Pickling
    #    pickle.dump(cor_listMaj, fp)
          
    hist = pickle.load( open( mypath + "/" + model + "_" + alpha + "_1histoChordAll.pkl", "rb" ) )
    all_listMaj = [0] * (len(dictChord)-1)
    all_listmin = [0] * (len(dictChord)-1)
    for key,value in hist.items():
        try:
            root, qual = key.split(":")
            for chord in range(len(dictChord)-1):
                root2, qual = listChord[chord].split(":")
                delRoot = delta_root(root,root2)%12
                if qual == "maj":
                    all_listmin[delRoot*2] += hist[key][chord]
                else:
                    all_listmin[delRoot*2+1] += hist[key][chord]
    
        except:
            if key == "N":
                root = key
                continue
            else:
                root = key
            for chord in range(len(dictChord)-1):
                root2, qual = listChord[chord].split(":")
                delRoot = delta_root(root,root2)%12
                if qual == "maj":
                    all_listMaj[delRoot*2] += hist[key][chord]
                else:
                    all_listMaj[delRoot*2+1] += hist[key][chord]     
    
    ratChord = [x/y if y else 0 for x,y in zip(cor_listMaj,all_listMaj)]
    print(all_listMaj)
    print(ratChord)
    #with open(mypath + "/" + model + "_" + alpha + "all.pkl", "wb" ) as fp:   #Pickling
    #    pickle.dump(all_listMaj, fp) 
    with open(mypath + "/histoCombine/" + model + "_" + alpha + "allmaj.pkl", "wb" ) as fp:   #Pickling
        pickle.dump(all_listMaj, fp)
    with open(mypath + "/histoCombine/" + model + "_" + alpha + "ratmaj.pkl", "wb" ) as fp:   #Pickling
        pickle.dump(ratChord, fp)

#%%
import random
import matplotlib.pyplot as plt
import numpy as np

minmaj = "maj"
model = "mlpDecim"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "all" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
basL = np.array(histDecim)
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decim = np.array(histDecim)
model = "mlpDecimFamily"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decimF = np.array(histDecim)
model = "mlpDecimKey"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decimK = np.array(histDecim)
model = "mlpDecimBeat"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decimB = np.array(histDecim)
model = "mlpDecimKeyBeat"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decimKB = np.array(histDecim)
model = "mlpDecimAug"
with open(mypath + "/histoCombine/" + model + "_" + alpha + "rat" + minmaj + ".pkl", "rb" ) as fp:   #Pickling
     histDecim = pickle.load(fp)
decimA = np.array(histDecim)

#%%
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.40
 
# set height of bar
listLeg = ['0-Maj','0-min','1-Maj','1-min','2-Maj','2-min','3-Maj','3-min', '4-Maj','4-min','5-Maj','5-min','6-Maj','6-min',
           '7-Maj','7-min','8-Maj','8-min','9-Maj','9-min','10-Maj','10-min','11-Maj','11-min']

#%%
toremove = []
listLeg = np.array(listLeg)
for i in range(len(basL)):
    if basL[i] == 0 : toremove.append(i-len(toremove))
#for listHist in [decim, decimK, decimB, decimKB, listLeg]:
def remZer(listHist, toremove):
    listHist = listHist.tolist()
    for torem in toremove:
        #listHist = listHist.tolist()
        listHist.pop(torem)
        #listHist = np.array(listHist)
    listHist = np.array(listHist)
    return listHist
basL = remZer(basL, toremove)
decim = remZer(decim, toremove)
decimF = remZer(decimF, toremove)
decimK = remZer(decimK, toremove)
decimB = remZer(decimB, toremove)
decimKB = remZer(decimKB, toremove)
decimA = remZer(decimA, toremove)


def produit(L1, L2):
    L3 = []
    for elt1, elt2 in zip(L1, L2):
        L3.append(elt1*elt2)
    return L3

'''
decim = produit(decim, basL)
decimF = produit(decimF, basL)
decimK = produit(decimK, basL)
decimB = produit(decimB, basL)
decimKB = produit(decimKB, basL)
decimA = produit(decimA, basL)
'''
listLeg = remZer(listLeg, toremove)
listLeg = listLeg.tolist()
#%%
for i in range(len(basL)):
    print(str(listLeg[i]) + " & " + str(int(basL[i])) + " & " + str(int(decim[i]))  + " & " +  str(int(decimK[i]))  + " & " +  str(int(decimB[i])) + " & " +  str(int(decimKB[i])) +"\\\\")
#%%
# Set position of bar on X axis
r5 = np.arange(len(listLeg))
r4 = [x + barWidth for x in r5]
r3 = [x + barWidth for x in r4]
r2 = [x + barWidth for x in r3]
r1 = [x + barWidth for x in r2]


# Make the plot
#plt.barh(r5, basL, color='dodgerblue', height=barWidth, edgecolor='white', label='Groundtruth')
plt.bar(r5, basL, color='midnightblue', width=barWidth, edgecolor='white', label='Groundtruth')
#plt.barh(r2, decim, color='green', height=barWidth, edgecolor='white', label='MLP-V')
#plt.barh(r3, decimK, color='red', height=barWidth, edgecolor='white', label='MLP-K')
#plt.barh(r4, decimB, color='blue', height=barWidth, edgecolor='white', label='MLP-B')
#plt.barh(r5, decimKB, color='purple', height=barWidth, edgecolor='white', label='MLP-KB')

plt.rcParams.update({'font.size': 22})
 
# Add xticks on the middle of the group bars
#plt.xlabel('On minor Key', fontweight='bold')
plt.xticks([r for r in range(len(listLeg))], listLeg,fontsize = 22,rotation=45)
plt.xlabel('valeurs des x')
#plt.xticks(0, "  ",rotation=90)
#plt.yscale('log')
# Create legend & Show graphic
#plt.legend()
#plt.gca().invert_yaxis()
plt.show()
plt.savefig("graphique.png")
#%%
fig = plt.gcf()
fig.set_size_inches(5, 2)
fig.savefig('test2png.pdf', dpi=100)
#%%
for i in range(len(basL)):
    print(str(listLeg[i]) + " & " + str(int(round(basL[i]/1000))) + "K & " + str(round(decim[i]/basL[i]*100,2))  + " & " +  str(round(decimK[i]/basL[i]*100,2))  + " & " +  str(round(decimB[i]/basL[i]*100,2)) + " & " +  str(round(decimKB[i]/basL[i]*100,2)) +"\\\\")

#%%
for i in range(len(basL)):
    print(str(listLeg[i]) + " & " + str(int(round(basL[i]/1000))) + "K & " + str(round(decim[i]*100,2))  + " & " + str(round(decimF[i]*100,2))  + " & " +  str(round(decimK[i]*100,2))  + " & " +  str(round(decimB[i]*100,2)) + " & " +  str(round(decimKB[i]*100,2))+ " & " + str(round(decimA[i]*100,2))  +"\\\\")