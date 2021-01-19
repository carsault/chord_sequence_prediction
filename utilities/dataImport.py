#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:45:20 2018

@author: carsault
"""

#%%
import torch
import torch.utils.data as data_utils
from random import randint
from torch.utils import data
from utilities import chordUtil
from utilities import distance
from utilities.distance import *
import numpy as np
import pickle
import os
import errno

def createDatasetFull(name):
    X = []
    y = []
    with open(name, 'rb') as pickle_file:
        test = pickle.load(pickle_file)
    return test

def saveSetDecim(listIDs, root, alpha, dictChord, listChord, dictChordGamme, gamme, lenSeq, lenPred, Decim, folder, part, lab = False, interv = False, asTensor = True, newRep = "non", maxReps = 1):
    Xfull = []
    yfull = []
    keyfull = []
    beatfull = []
    dictDat = {}
    '''
    if newRep == "newRep":
        file2 = open("testNewrep/testfileNEWX.txt","w")
        fileY2 = open("testNewrep/testfileNEWY.txt","w") 
    else:
        file = open("testNewrep/testfileNoramX.txt","w")
        fileY = open("testNewrep/testfileNoramY.txt","w")
    '''
    try:
        os.mkdir(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    minorKey = 0
    for track in listIDs:
        #print(track)
        beatInf = []
        # Open xlab
        xlab = open(root + track,"r")
        lines = xlab.read().split("\n")
        # Transform with one chord by beat
        chordBeat = []
        key = []
        # Initialize with N
        for i in range(lenSeq-1):
            beatInf.append(4) #if it's before start downbeat information is 4
            chordBeat.append(dictChord[chordUtil.reduChord('N', alpha)])
            key.append(dictChordGamme['N'])
        # Complete with chords in the file
        nbMaxChordFollow = 0
        currChordFollow = 0
        lastChord = "NUL"
        for i in range(len(lines)-1):
            line = lines[i+1].split(" ")
            downBeat = line[0].split(":")
            for j in range(int(line[2])):
                if alpha == "functional":
                    currChord = dictChord[chordUtil.reduChord(line[4], "a0")]
                else:
                    currChord = dictChord[chordUtil.reduChord(line[4], alpha)]
                if lastChord == currChord:
                    currChordFollow += 1
                else:
                    if currChordFollow > nbMaxChordFollow:
                        nbMaxChordFollow = currChordFollow
                    currChordFollow = 0
                beatInf.append((int(downBeat[1])+j-1)%4) #get beat minus one, times j  
                chordBeat.append(currChord)
                key.append(dictChordGamme[gamme[line[6]]])
                lastChord = currChord
        if len(set(chordBeat)) == 2:
            print(track)
            print(set(chordBeat))
        if nbMaxChordFollow > 31:
            print(track)
        # Iterate over the track
        for start in range(len(chordBeat)-lenPred-lenSeq+1):
            if lab == False:
                if interv:
                    X = torch.zeros(lenSeq, 12)
                else:
                    X = torch.zeros(lenSeq, len(listChord))
            
                for i in range(lenSeq):
                    if interv:
                        if listChord[chordBeat[start+i]] is not 'N':
                            X[i] = torch.FloatTensor(representation(listChord[chordBeat[start+i]], 0.5))
                    else:
                        X[i][chordBeat[start+i]] = 1
            else:
                X = torch.zeros(lenSeq)
                for i in range(lenSeq):
                    X[i] = chordBeat[start+i]
            # Get label
            if lab == False:
                y = torch.zeros(lenPred, len(listChord))
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    y[i][chordBeat[start+lenSeq+i]] = 1
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
            else:
                if newRep == "newRep":
                    y = []
                else:
                    y = torch.zeros(lenPred)
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                if newRep == "newRep":
                    for i in range(lenPred+maxReps):
                        if start+lenSeq+i < len(chordBeat):
                            if start+lenSeq+i-1>0:
                                if i < lenPred-1:
                                    y.append(chordBeat[start+lenSeq+i])
                                else:
                                    if chordBeat[start+lenSeq+i] == chordBeat[start+lenSeq+i-1]:
                                        y.append(chordBeat[start+lenSeq+i])
                        else:
                            y.append(len(listChord))
                    y = torch.Tensor(y)
                else:
                    for i in range(lenPred):
                        y[i] = chordBeat[start+lenSeq+i]
                    
            listX = []
            listy = []
            if newRep != "newRep":
                for i in Decim:
                    u = []
                    decimX = torch.chunk(X, int(lenSeq / i))
                    for j in range(len(decimX)):
                        u.append(torch.sum(decimX[j], 0))
                    u = torch.stack(u)
                    listX.append(u)
                    u = []
                    decimy = torch.chunk(y, int(lenPred / i))
                    for j in range(len(decimy)):
                        u.append(torch.sum(decimy[j], 0))
                    u = torch.stack(u)
                    listy.append(u)
                listX = torch.cat(listX, 0)
                listy = torch.cat(listy, 0)
            if newRep == "newRep":
                listy = y
                listX = X
                listX = chordUtil.transFromRep(listX,maxReps,len(dictChord),lenSeq)
                listy = chordUtil.transFromRepY(listy,maxReps,len(dictChord),lenPred)
                #listX = torch.Tensor(listX)
                #listy = torch.Tensor(listy)
                #listX, listy = chordUtil.transfToPrevRepNoProb(listX,listy, maxReps,len(dictChord), False)
                #print(listy)
                listX = torch.Tensor(listX)
                listy = torch.Tensor(listy)
                #print(listX)
                '''
                file2.write(str(listX))
                file2.write("\n")
                fileY2.write(str(listy))
                fileY2.write("\n")
            else:
                file.write(str(listX))
                file.write("\n")
                fileY.write(str(listy))
                fileY.write("\n")
                '''   
            Xfull.append(listX)
            yfull.append(listy)
            beatfull.append(numBeat)
            keyfull.append(localKey)
    '''        
    if newRep == "newRep":
        file2.close()
        fileY2.close() 
    else:
        file.close()
        fileY.close()
    '''
       
    Xfull = torch.stack(Xfull)
    yfull = torch.stack(yfull)
    keyfull = torch.stack(keyfull)
    beatfull = torch.stack(beatfull)

    if asTensor == True:
        sauv = open(folder + '/' + part +".pkl","wb")  
        #pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv, protocol=4)
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
    else:
        dictDat["X"] = Xfull
        dictDat["y"] = yfull
        dictDat["key"] = keyfull
        dictDat["beat"] = beatfull
        sauv = open(folder + '/' + part + ".pkl","wb")  
        pickle.dump(dictDat,sauv)

    sauv.close()
    print("number of minor songs in this dataset:" + str(minorKey))

def saveSetDecimNewRepGOOD(listIDs, root, alpha, dictChord, listChord, newdictChord, newlistChord, maxRep, dictChordGamme, gamme, lenSeq, lenPred, Decim, folder, part, lab = False, interv = False, groupBy = True):
    Xfull = []
    yfull = []
    keyfull = []
    beatfull = []
    dictDat = {}
    try:
        os.mkdir(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    minorKey = 0
    for track in listIDs:
        #print(track)
        beatInf = []
        # Open xlab
        xlab = open(root + track,"r")
        lines = xlab.read().split("\n")
        # Transform with one chord by beat
        chordBeat = []
        key = []
        # Initialize with N
        for i in range(lenSeq-1):
            beatInf.append(4) #if it's before start downbeat information is 4
            chordBeat.append(dictChord[chordUtil.reduChord('N', alpha)])
            key.append(dictChordGamme['N'])
        # Complete with chords in the file
        '''
        newRep = []
        newRepBeat = []
        interRep = ''
        interBeat = 0
        for i in len(chordBeat):
            if i = 0:
                interRep = chordBeat[0]
                interBeat  += 1
            else:
                if interRep == chordBeat[i]:
                    interBeat  += 1
                else:
                    newRep.append(interRep)
                    newBeat.append(interBeat)
        '''
        nbMaxChordFollow = 0
        currChordFollow = 0
        lastChord = "NUL"
        #get maximum same chord sequence
        for i in range(len(lines)-1):
            line = lines[i+1].split(" ")
            downBeat = line[0].split(":")
            for j in range(int(line[2])):
                if alpha == "functional":
                    currChord = dictChord[chordUtil.reduChord(line[4], "a0")]
                else:
                    currChord = dictChord[chordUtil.reduChord(line[4], alpha)]
                if lastChord == currChord:
                    currChordFollow += 1
                else:
                    if currChordFollow > nbMaxChordFollow:
                        nbMaxChordFollow = currChordFollow
                    currChordFollow = 0
                beatInf.append((int(downBeat[1])+j-1)%4) #get beat minus one, times j  
                chordBeat.append(currChord)
                key.append(dictChordGamme[gamme[line[6]]])
                lastChord = currChord
        if len(set(chordBeat)) == 2:
            print(track)
            print(set(chordBeat))
        if nbMaxChordFollow > 31:
            print(track)
        # Iterate over the track
        for start in range(len(chordBeat)-lenPred-lenSeq+1):
            if lab == False:
                if interv:
                    X = torch.zeros(lenSeq, 12)
                else:
                    X = torch.zeros(lenSeq, len(listChord))
                newRep = []
                newBeat = []
                interRep = ""
                interBeat = 0
                for i in range(lenSeq):
                    if interv:
                        if listChord[chordBeat[start+i]] is not 'N':
                            X[i] = torch.FloatTensor(representation(listChord[chordBeat[start+i]], 0.5))
                    else:
                        if groupBy == True:
                            if i == 0:
                                interRep = chordBeat[start]
                                interBeat  += 1
                            elif i == lenSeq-1:
                                if interRep == chordBeat[start+i]:
                                    interBeat  += 1
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                else:
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                    interRep = chordBeat[start+i]
                                    interBeat = 1
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                            else:
                                if interRep == chordBeat[start+i]:
                                    interBeat  += 1
                                else:
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                    interRep = chordBeat[start+i]
                                    interBeat = 1
                        else:
                            X[i][chordBeat[start+i]] = 1
                #print("new rep x :" + str(newRep))
                #print("new beat x :" + str(newBeat))
                if groupBy == True:
                    X = torch.zeros(lenSeq, len(newlistChord))
                    numLigne = 0
                    for i in range(len(newRep)):
                        while newBeat[i] > 0 and numLigne < 8:
                            if (newBeat[i] % maxRep) != 0:
                                X[numLigne][newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]] = 1
                                newBeat[i] -=  newBeat[i] % maxRep
                                numLigne += 1
                            else:
                                X[numLigne][newdictChord[listChord[newRep[i]] + " " + str(maxRep)]] = 1
                                newBeat[i] -=  maxRep
                                numLigne += 1
                    for i in range(lenSeq-len(newRep)-1):
                        X[i+len(newRep)+1][newdictChord["N"]] = 1
            else:
                X = torch.zeros(lenSeq)
                for i in range(lenSeq):
                    X[i] = chordBeat[start+i]
            newRep = []
            newBeat = []
            interRep = ""
            interBeat = 0
            # Get label
            if lab == False:
                if groupBy == True:
                    if (len(chordBeat)-(start+lenSeq+lenPred))>=2:
                        extra = 2
                    elif (len(chordBeat)-(start+lenSeq+lenPred))==1:
                        extra = 1
                    else:
                        extra = 0
                    for i in range(lenPred+extra):
                        if i == 0:
                            interRep = chordBeat[start+lenSeq]
                            interBeat  += 1
                        elif i == lenPred+maxRep-1:
                            if interRep == chordBeat[start+lenSeq+i]:
                                interBeat  += 1
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                            else:
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                                interRep = chordBeat[start+lenSeq+i]
                                interBeat = 1
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                        else:
                            if interRep == chordBeat[start+lenSeq+i]:
                                interBeat  += 1
                            else:
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                                interRep = chordBeat[start+lenSeq+i]
                                interBeat = 1
                    for i in range(len(newBeat)):
                        if sum(newBeat[:-1])>=lenPred:
                            newBeat.pop()
                            newRep.pop()
                else:
                    y = torch.zeros(lenPred, len(listChord))
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    if groupBy != True: y[i][chordBeat[start+lenSeq+i]] = 1
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                if groupBy == True:
                    y = torch.zeros(lenSeq, len(newlistChord))
                    #print("new rep y :" + str(newRep))
                    #print("new beat y :" + str(newBeat))
                    numLigne = 0
                    for i in range(len(newRep)):    
                        while newBeat[i] > 0 and numLigne < 8:
                            #print(newBeat[i])
                            if (newBeat[i] % maxRep) != 0:
                                y[numLigne][newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]] = 1
                                #print(newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)])
                                newBeat[i] -=  newBeat[i] % maxRep
                                numLigne += 1
                            else:
                                y[numLigne][newdictChord[listChord[newRep[i]] + " " + str(maxRep)]] = 1
                                #print(newdictChord[listChord[newRep[i]] + " " + str(maxRep)])
                                newBeat[i] -=  maxRep
                                numLigne += 1
                    for i in range(lenSeq-len(newRep)-1):
                        y[i+len(newRep)+1][newdictChord["N"]] = 1
                    #print(y)
            else:
                y = torch.zeros(lenPred)
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    y[i] = chordBeat[start+lenSeq+i]
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                    
            listX = []
            listy = []
            for i in Decim:
                u = []
                decimX = torch.chunk(X, int(lenSeq / i))
                for j in range(len(decimX)):
                    u.append(torch.sum(decimX[j], 0))
                u = torch.stack(u)
                listX.append(u)
                u = []
                decimy = torch.chunk(y, int(lenPred / i))
                for j in range(len(decimy)):
                    u.append(torch.sum(decimy[j], 0))
                u = torch.stack(u)
                listy.append(u)
            listX = torch.cat(listX, 0)
            listy = torch.cat(listy, 0)
            print(listX)
            Xfull.append(listX)
            yfull.append(listy)
            beatfull.append(numBeat)
            keyfull.append(localKey)
            
    Xfull = torch.stack(Xfull)
    yfull = torch.stack(yfull)
    keyfull = torch.stack(keyfull)
    beatfull = torch.stack(beatfull)

    if lab == False:
        sauv = open(folder + '/' + part +".pkl","wb")  
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
    else:
        dictDat["X"] = Xfull
        dictDat["y"] = yfull
        dictDat["key"] = keyfull
        dictDat["beat"] = beatfull
        sauv = open(folder + '/' + part + ".pkl","wb")  
        pickle.dump(dictDat,sauv)

    sauv.close()
    print("number of minor songs in this dataset:" + str(minorKey))

def saveSetDecimNewRep(listIDs, root, alpha, dictChord, listChord, newdictChord, newlistChord, maxRep, dictChordGamme, gamme, lenSeq, lenPred, Decim, folder, part, lab = False, interv = False, groupBy = True):
    Xfull = []
    yfull = []
    keyfull = []
    beatfull = []
    dictDat = {}
    try:
        os.mkdir(folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    minorKey = 0
    for track in listIDs:
        #print(track)
        beatInf = []
        # Open xlab
        xlab = open(root + track,"r")
        lines = xlab.read().split("\n")
        # Transform with one chord by beat
        chordBeat = []
        key = []
        # Initialize with N
        for i in range(lenSeq-1):
            beatInf.append(4) #if it's before start downbeat information is 4
            chordBeat.append(dictChord[chordUtil.reduChord('N', alpha)])
            key.append(dictChordGamme['N'])
        # Complete with chords in the file
        '''
        newRep = []
        newRepBeat = []
        interRep = ''
        interBeat = 0
        for i in len(chordBeat):
            if i = 0:
                interRep = chordBeat[0]
                interBeat  += 1
            else:
                if interRep == chordBeat[i]:
                    interBeat  += 1
                else:
                    newRep.append(interRep)
                    newBeat.append(interBeat)
        '''
        nbMaxChordFollow = 0
        currChordFollow = 0
        lastChord = "NUL"
        #get maximum same chord sequence
        for i in range(len(lines)-1):
            line = lines[i+1].split(" ")
            downBeat = line[0].split(":")
            for j in range(int(line[2])):
                if alpha == "functional":
                    currChord = dictChord[chordUtil.reduChord(line[4], "a0")]
                else:
                    currChord = dictChord[chordUtil.reduChord(line[4], alpha)]
                if lastChord == currChord:
                    currChordFollow += 1
                else:
                    if currChordFollow > nbMaxChordFollow:
                        nbMaxChordFollow = currChordFollow
                    currChordFollow = 0
                beatInf.append((int(downBeat[1])+j-1)%4) #get beat minus one, times j  
                chordBeat.append(currChord)
                key.append(dictChordGamme[gamme[line[6]]])
                lastChord = currChord
        if len(set(chordBeat)) == 2:
            print(track)
            print(set(chordBeat))
        if nbMaxChordFollow > 31:
            print(track)
        # Iterate over the track
        for start in range(len(chordBeat)-lenPred-lenSeq+1):
            if lab == False:
                if interv:
                    X = torch.zeros(lenSeq, 12)
                else:
                    X = torch.zeros(lenSeq, len(listChord))
                newRep = []
                newBeat = []
                interRep = ""
                interBeat = 0
                for i in range(lenSeq):
                    if interv:
                        if listChord[chordBeat[start+i]] is not 'N':
                            X[i] = torch.FloatTensor(representation(listChord[chordBeat[start+i]], 0.5))
                    else:
                        if groupBy == True:
                            if i == 0:
                                interRep = chordBeat[start]
                                interBeat  += 1
                            elif i == lenSeq-1:
                                if interRep == chordBeat[start+i]:
                                    interBeat  += 1
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                else:
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                    interRep = chordBeat[start+i]
                                    interBeat = 1
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                            else:
                                if interRep == chordBeat[start+i]:
                                    interBeat  += 1
                                else:
                                    newRep.append(interRep)
                                    newBeat.append(interBeat)
                                    interRep = chordBeat[start+i]
                                    interBeat = 1
                        else:
                            X[i][chordBeat[start+i]] = 1
                #print("new rep x :" + str(newRep))
                #print("new beat x :" + str(newBeat))
                if groupBy == True:
                    #X = torch.zeros(lenSeq, len(newlistChord))
                    X = torch.zeros(lenSeq)
                    numLigne = 0
                    for i in range(len(newRep)):
                        while newBeat[i] > 0 and numLigne < 8:
                            if (newBeat[i] % maxRep) != 0:
                                #X[numLigne][newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]] = 1
                                X[numLigne] = newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]
                                newBeat[i] -=  newBeat[i] % maxRep
                                numLigne += 1
                            else:
                                #X[numLigne][newdictChord[listChord[newRep[i]] + " " + str(maxRep)]] = 1
                                X[numLigne] = newdictChord[listChord[newRep[i]] + " " + str(maxRep)]
                                newBeat[i] -=  maxRep
                                numLigne += 1
                    for i in range(lenSeq-len(newRep)-1):
                        #X[i+len(newRep)+1][newdictChord["N"]] = 1
                        X[i+len(newRep)+1] = newdictChord["N"]
            else:
                X = torch.zeros(lenSeq)
                for i in range(lenSeq):
                    X[i] = chordBeat[start+i]
            newRep = []
            newBeat = []
            interRep = ""
            interBeat = 0
            # Get label
            if lab == False:
                if groupBy == True:
                    if (len(chordBeat)-(start+lenSeq+lenPred))>=2:
                        extra = 2
                    elif (len(chordBeat)-(start+lenSeq+lenPred))==1:
                        extra = 1
                    else:
                        extra = 0
                    for i in range(lenPred+extra):
                        if i == 0:
                            interRep = chordBeat[start+lenSeq]
                            interBeat  += 1
                        elif i == lenPred+maxRep-1:
                            if interRep == chordBeat[start+lenSeq+i]:
                                interBeat  += 1
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                            else:
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                                interRep = chordBeat[start+lenSeq+i]
                                interBeat = 1
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                        else:
                            if interRep == chordBeat[start+lenSeq+i]:
                                interBeat  += 1
                            else:
                                newRep.append(interRep)
                                newBeat.append(interBeat)
                                interRep = chordBeat[start+lenSeq+i]
                                interBeat = 1
                    for i in range(len(newBeat)):
                        if sum(newBeat[:-1])>=lenPred:
                            newBeat.pop()
                            newRep.pop()
                else:
                    y = torch.zeros(lenPred, len(listChord))
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    if groupBy != True: y[i][chordBeat[start+lenSeq+i]] = 1
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                if groupBy == True:
                    #y = torch.zeros(lenSeq, len(newlistChord))
                    y = torch.zeros(lenSeq)
                    #print("new rep y :" + str(newRep))
                    #print("new beat y :" + str(newBeat))
                    numLigne = 0
                    for i in range(len(newRep)):    
                        while newBeat[i] > 0 and numLigne < 8:
                            #print(newBeat[i])
                            if (newBeat[i] % maxRep) != 0:
                                #y[numLigne][newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]] = 1
                                y[numLigne] = newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)]
                                #print(newdictChord[listChord[newRep[i]] + " " + str(newBeat[i] % maxRep)])
                                newBeat[i] -=  newBeat[i] % maxRep
                                numLigne += 1
                            else:
                                #y[numLigne][newdictChord[listChord[newRep[i]] + " " + str(maxRep)]] = 1
                                y[numLigne] = newdictChord[listChord[newRep[i]] + " " + str(maxRep)]
                                #print(newdictChord[listChord[newRep[i]] + " " + str(maxRep)])
                                newBeat[i] -=  maxRep
                                numLigne += 1
                    for i in range(lenSeq-len(newRep)-1):
                        #y[i+len(newRep)+1][newdictChord["N"]] = 1
                        y[i+len(newRep)+1] = newdictChord["N"]
                    #print(y)
            else:
                y = torch.zeros(lenPred)
                numBeat = torch.zeros(lenPred)
                localKey = torch.zeros(lenPred)
                for i in range(lenPred):
                    y[i] = chordBeat[start+lenSeq+i]
                    numBeat[i] = beatInf[start+lenSeq+i]
                    localKey[i] = key[start+lenSeq+i]
                    
            listX = []
            listy = []
            for i in Decim:
                u = []
                decimX = torch.chunk(X, int(lenSeq / i))
                for j in range(len(decimX)):
                    u.append(torch.sum(decimX[j], 0))
                u = torch.stack(u)
                listX.append(u)
                u = []
                decimy = torch.chunk(y, int(lenPred / i))
                for j in range(len(decimy)):
                    u.append(torch.sum(decimy[j], 0))
                u = torch.stack(u)
                listy.append(u)
            listX = torch.cat(listX, 0)
            listy = torch.cat(listy, 0)
            Xfull.append(listX)
            yfull.append(listy)
            beatfull.append(numBeat)
            keyfull.append(localKey)       
    Xfull = torch.stack(Xfull)
    yfull = torch.stack(yfull)
    keyfull = torch.stack(keyfull)
    beatfull = torch.stack(beatfull)

    if lab == False:
        sauv = open(folder + '/' + part +".pkl","wb")  
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
        pickle.dump(data_utils.TensorDataset(Xfull, yfull, keyfull, beatfull),sauv)
    else:
        dictDat["X"] = Xfull
        dictDat["y"] = yfull
        dictDat["key"] = keyfull
        dictDat["beat"] = beatfull
        sauv = open(folder + '/' + part + ".pkl","wb")  
        pickle.dump(dictDat,sauv)

    sauv.close()
    print("number of minor songs in this dataset:" + str(minorKey))
