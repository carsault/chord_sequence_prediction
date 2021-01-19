#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:38:02 2017

@author: tristan
"""

"""----------------------------------------------------------------------
-- Tristan Metadata and conv
----------------------------------------------------------------------"""

#%%
from utilities.chordVocab import *
#import torch

def getDictChord(alpha):
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''
    chordList = []
    dictChord = {}
    for v in gamme.values():
        if v != 'N':
            for u in alpha.values():
                if u != 'N':
                    chordList.append(v+":"+u)
    chordList.append('N')
    listChord = list(set(chordList))
    listChord.sort()
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    return dictChord, listChord

def getDictKey():
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''
    chordList = []
    dictChord = {}
    for v in gammeKey.values():
        chordList.append(v)
    chordList.append('N')
    listChord = list(set(chordList))
    listChord.sort()
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    return dictChord, listChord

#dictA0 = getDictChord(a3)
    
def getDictChordUpgrade(alpha, max_len):
    chordList = []
    dictChord = {}
    for v in gamme.values():
        if v != 'N':
            for u in alpha.values():
                if u != 'N':
                    for w in range(max_len):
                        beat_w = w+1
                        chordList.append(str(v)+":"+str(u)+" " + str(beat_w))
    for w in range(max_len):
        chordList.append('N' + " " + str(w+1))
    listChord = list(set(chordList))
    listChord.sort()
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
    #for i in range(1, max_len+1):
    #    dictChord.update({'N'+ " " + str(i): len(dictChord)})
    #    listChord.append('N'+ " " + str(i))
    #print(dictChord)
    return dictChord, listChord

def computeAccNewRep(args,decim,output,local_labels):
    totalTest = 0
    correct = 0
    lablbatmax = local_labels.data.max(2, keepdim=False)[1]
    #print(lablbatmax)
    for i in range(output.size()[0]):
        oldRepOut = []
        oldRepPred = []
        #print(local_labels)
        for j in range(int(args.lenPred/decim)):
        #retreive old rep for output
            valChord = output[i][j].max(0)[1]
            #print(valChord)
            chord = valChord//args.maxReps
            lenBeat = valChord%args.maxReps+1
            #print(chord)
            #print(lenBeat)
            for m in range(lenBeat):
                oldRepOut.append(chord)
            #retreive old rep for pred
            valChord = local_labels[i][j].max(0)[1]
            #print(valChord)
            #print(valChord)
            chord = valChord//args.maxReps
            lenBeat = valChord%args.maxReps+1
            for m in range(lenBeat):
                oldRepPred.append(chord)
           # print(oldRepPred)
        #Keep only the first 8 elements
        oldRepOut = oldRepOut[0:args.lenPred]
        oldRepPred = oldRepPred[0:args.lenPred]
        #print(oldRepOut)
        #print(oldRepPred)
        #Do the comparison
        #print(len(oldRepOut))
        for j in range(int(args.lenPred/decim)):
            totalTest += 1
            result = (oldRepOut[j] == oldRepPred[j]).item()
            correct += result
    return correct, totalTest

def transfToPrevRep(args,output,local_labels, prob = True):
#def transfToPrevRep(output,local_labels, maxReps,n_categories, prob = True):
    #for i in range(output.size()[0]):
    oldRepOut = []
    oldRepLab = []
    #print(local_labels)
    #for j in range(int(args.lenPred)):
    for j in range(len(output)):
    #retreive old rep for output
        if prob:
            valChord = output[j].max(0)[1]
        else:
            valChord = output[j]
        #print(valChord)
        #if valChord > (args.n_categories - 1):
        #    valChord -= 1
        chord = valChord//args.maxReps
        lenBeat = valChord%args.maxReps+1
        #print(chord)
        #print(lenBeat)
        #for m in range(int(lenBeat)):
        for m in range(lenBeat):
            oldRepOut.append(chord)
        #retreive old rep for pred
        if prob:
            valChord = local_labels[j].max(0)[1]
        else:
            valChord = local_labels[j]
        #print(valChord)
        #print(valChord)
        #if valChord > (n_categories - 1):
        #    valChord -= 1
        chord = valChord//args.maxReps
        lenBeat = valChord%args.maxReps+1
        #for m in range(int(lenBeat)):
        for m in range(lenBeat):
            oldRepLab.append(chord)
       # print(oldRepPred)
    #Keep only the first 8 elements
    oldRepOut = oldRepOut[0:len(output)]
    
    #if we measure X
    #oldRepOut.reverse()
    
    #oldRepLab = oldRepLab[-len(output):-1]
    oldRepLab = oldRepLab[0:len(output)]
    return oldRepOut, oldRepLab


def transFromRep(listX,maxReps,n_categories,lenPred):
    numRep = 0
    curChord = listX[len(listX)-1].item()
    newList = []
    for i in range(len(listX)-1):
        if listX[len(listX)-2-i] == curChord:
            numRep += 1
            if numRep > maxReps-1:
                newList.append(curChord*maxReps-1+numRep)
                curChord = listX[len(listX)-2-i].item()
                numRep = 0
        else:
            newList.append(curChord*maxReps+numRep)
            curChord = listX[len(listX)-2-i].item()
            numRep = 0
    newList.append(curChord*maxReps+numRep)
    #for i in range(len(listX)-len(newList)):
    for i in range(lenPred-len(newList)):
        newList.append((n_categories-1)*maxReps)
    #print(newList)
    return newList

def transFromRepY(listX,maxReps,n_categories,lenPred):
    numRep = 0
    curChord = listX[0].item()
    newList = []
    nbMAxpred = 1
    initBeat = 1
    initChord = listX[0]
    firstOk = False
    for i in range(len(listX)-1):
        #check if we stop to fill
        if nbMAxpred > lenPred and listX[1+i] != curChord:
            break
        #check if we stop to fill
        if nbMAxpred >= lenPred and numRep == maxReps - 1:
            break
        #reverse the first
        if listX[1+i] == initChord:
            initBeat += 1
            if i == 0:
                firstOk = True
        #check if this is the same
        if listX[1+i] == curChord:
            numRep += 1
            #if we are over the max we save
            if numRep > maxReps-1:
                newList.append(curChord*maxReps-1+numRep)
                curChord = listX[1+i].item()
                numRep = 0
        #if this is different we save the last
        else:
            newList.append(curChord*maxReps+numRep)
            curChord = listX[1+i].item()
            numRep = 0
        nbMAxpred +=1
    if curChord*maxReps+numRep> (n_categories-1)*maxReps:
        #print(curChord*maxReps+numRep)
        #print(newList)
        newList.append((n_categories-1)*maxReps)
    else:
        newList.append(curChord*maxReps+numRep)
    #for i in range(lenPred-len(newList)):
    for i in range(lenPred):
        newList.append((n_categories-1)*maxReps)
    #print(newList)
    if initBeat%2 != 0 and firstOk:
        u = newList[0]
        newList[0] = newList[1]
        newList[1] = u
    newList = newList[0:lenPred]
    return newList

def reduChord(initChord, alpha= 'a1', transp = 0):
    '''
    Fonction def

    Parameters
    ----------
    tf_mapping: keras.backend tensor float32
        mapping of the costs for the loss function

    Returns
    -------
    loss_function: function
    '''    
    if initChord == "":
        print("buuug")
    initChord, bass = initChord.split("/") if "/" in initChord else (initChord, "")
    root, qual = initChord.split(":") if ":" in initChord else (initChord, "")
    root, noChord = root.split("(") if "(" in root else (root, "")
    qual, additionalNotes = qual.split("(") if "(" in qual else (qual, "")  
    
    root = gamme[root]
    if transp > 0:
        for i in range(transp):
            root = tr[root]
    else:
        for i in range(12+transp):
            #print("transpo")
            root = tr[root]
    
    if qual == "":
        if root == "N" or noChord != "":
            finalChord = "N"
        else:
            finalChord = root + ':maj'
    
    elif root == "N":
        finalChord = "N"
    
    else:
        if alpha == 'a1':
                qual = a1[qual]
        elif alpha == 'a0':
                qual = a0[qual]
        elif alpha == 'a2':
                qual = a2[qual]
        elif alpha == 'a3':
                qual = a3[qual]
        elif alpha == 'a5':
                qual = a5[qual]
        elif alpha == 'reduceWOmodif':
                qual = qual
        else:
                print("wrong alphabet value")
                qual = qual
        if qual == "N":
            finalChord = "N"
        else:
            finalChord = root + ':' + qual

    return finalChord
