#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:51:53 2019

@author: carsault
"""

#%%
import pickle
import torch
from utilities import chordUtil
from utilities.chordUtil import *
from utilities import testFunc
from utilities.testFunc import *
from utilities import distance
from utilities.distance import *
from ACE_Analyzer import ACEAnalyzer
from ACE_Analyzer.ACEAnalyzer import *
import numpy as np
import time

#time.sleep(3600)

# CUDA for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor

foldName = "modelSave200908"
#modelType = ["mlpDecim", "mlpDecimKey", "mlpDecimBeat", "mlpDecimKeyBeat","mlpDecimAug","mlpDecimFamily"]
#modelType = ["lstmDecim"]
modelType = ["mlpDecim", "mlpDecimKey", "mlpDecimBeat", "mlpDecimKeyBeat","mlpDecimAug","lstmDecim","mlpDecimFamily"]
randm = [1, 2, 3, 4, 5]
#alpha = ['a0','a2','a5','functional']
alpha = ['a0','a2','a5']
maxRepList = [1,2,4,8]
alphaRep = ["Nope","alphaRep"]
dictKey, listKey = chordUtil.getDictKey()
correctDown = [0]*4
correctPos = [0]*8
totalDown = [0]*4
totalPos = [0]*8
musicalDist2 = 0
musicalDist4 = 0
dictFinalRes = {}
specName = ""

#test
#modelType = ["mlpDecim"]
#foldName = "modelSave190515"
#randm = [1]
#alpha = ['a0']
#%%
def perplexityAccumulate(res, y, sPrev, nby ,one_hot = False):
    s = 0
    oneHot = []
    if one_hot == True:
        for i in range(len(y)):
            oneHot.append(y[i].index(max(y[i])))
    else:
        oneHot = y
    for r, t in zip(res,oneHot):
        s += np.log2(r[t])

    sPrev += s
    nby += len(y)
    return sPrev, nby

def perplexityCompute(s,y):
    s /= -y
    return 2 ** s

#%%
def perplexity(res, y, one_hot = False):
    s = 0
    oneHot = []
    if one_hot == True:
        for i in range(len(y)):
            oneHot.append(y[i].index(max(y[i])))
    else:
        oneHot = y
    for r, t in zip(res,oneHot):
        s += np.log2(r[t])

    s /= -len(y)
    return 2 ** s
#%%
for alph in alpha:
    for model in modelType:
        if model == "mlpDecimFamily":
            str1 = "124"
        else:
            str1 = "1"
        for maxReps in maxRepList:
            if maxReps == 1:
                alphaRep = "Nope"
            else:
                alphaRep = "alphaRep"
            #if maxReps == 1 or model == "mlpDecim" or model == "mlpDecimAug" or model == "lstmDecim":
            if maxReps == 1 or model == "mlpDecim" or model == "mlpDecimAug":
                print("\n------\n------\nResult for " + model + "on alphabet " + alph + "\n------\n------\n")
                perp = []
                rank = []
                totRank = []
                #Analyzer = ACEAnalyzer()
                
                if alph != "functional":  
                    dictChord, listChord = chordUtil.getDictChord(eval(alph))
                    distMat = testFunc.computeMat(dictChord, "euclidian")
                    #if alphaRep == "alphaRep":
                    #    dictChord, listChord = chordUtil.getDictChordUpgrade(eval(alph), maxReps)
                    dictKeyChord = {}
                    dictKeyChordAll = {}
                    for key in dictKey:
                        dictKeyChord[key] = np.zeros(len(listChord))
                        dictKeyChordAll[key] = np.zeros(len(listChord))
                    Analyzer = ACEAnalyzer()
                    AnalyzerDiat = ACEAnalyzer()
                    AnalyzerNoDiat = ACEAnalyzer()
                    
                if alph == "a0":
                    dictFun = relatif
                
        
                totAcc = []
                totAccRepeat = []
                totAccDiat = []
                totAccNoDiat = []
                totKey= []
                totDownbeat = []
                totAccFun = []
                totAcca0 = []
                totAccDiat = []
                totAccNoDiat = []
                totAccDownbeat = []
                totAccPos = []
                totAccBeatPos = []
                totDist = []
                predTotDist = []
                totDistPred = []
                totDist2a = []
                totDist4a = []        
                totDist2b = []
                totDist4b = []
                totDist2c = []
                totDist4c = []
                nbCorrectChordDiat = 0
                nbCorrectChordNoDiat = 0
                nbTotalDiat = 0
                nbTotalNoDiat = 0
                for rand in randm:
                    '''
                    sumPred2ab = np.zeros(len(listChord))
                    sumTarg2ab = np.zeros(len(listChord))
                    sumPred4ab = np.zeros(len(listChord))
                    sumTarg4ab = np.zeros(len(listChord))
                    sumPred2c = np.zeros(len(listChord))
                    sumTarg2c = np.zeros(len(listChord))
                    sumPred4c = np.zeros(len(listChord))
                    sumTarg4c = np.zeros(len(listChord))
                    '''
                    correctDown = [0]*4
                    correctPos = [0]*8
                    totalDown = [0]*4
                    totalPos = [0]*8
                    correctBeatPos = np.zeros((4,8))
                    totalBeatPos = np.zeros((4,8))
                    accBeatPos = np.zeros((4,8))
                    musicalDist = 0
                    predMusicalDist = 0
                    acc2correct = 0
                    acc4correct = 0
                    musicalDist2a = 0
                    musicalDist4a = 0
                    musicalDist2b = 0
                    musicalDist4b = 0
                    musicalDist2c = 0
                    musicalDist4c = 0
                    zerr = np.zeros(len(dictChord))
                    total = 0
                    totalRepeat = 0
                    totalDiat = 0
                    totalNoDiat = 0
                    acc = 0
                    accRepeat = 0
                    accDiat = 0
                    accNoDiat = 0
                    correct = 0
                    correctDiat = 0
                    correctNoDiat = 0
                    keycorrect = 0
                    downbeatcorrect = 0
                    correctReducFunc = 0
                    correctReduca0 = 0
                    #dataFolder = alph + "_1_" + str(rand)
        
                    if alphaRep == "alphaRep":
                        dataFolder = alph + "_1_" + str(rand) + "newRep" + str(maxReps)
                    else:
                        dataFolder = alph + "_124_" + str(rand)
                    
                    #modelName = dataFolder + "_" + str1 + "_" + model + specName
                    if modelType == "mlpDecimFamily":
                        modelName = dataFolder + "_124_" + model  + specName
                    else:
                        modelName = dataFolder + "_" + str1 + "_" + model  + specName
                    with open("testVector/" + foldName + '/' + modelName + '/' + "probVect_" + modelName + "_test.pkl", 'rb') as fp:
                        dictDat = pickle.load(fp)
                        dictDat["X"] = dictDat["X"].cpu().numpy()
                        dictDat["y"] = dictDat["y"].cpu().numpy()
                        
                    if alph != "functional": 
                        #musicalDist = np.sum(np.matmul(dictDat["X"], distMat) * dictDat["y"])
                        musicalDist = 0
                    for i in range(len(dictDat["X"])):
        
                        pred = np.argmax(dictDat["X"][i])
                        tgt = np.argmax(dictDat["y"][i])
                        # rank of the chords
                        rank.append(len(np.where(dictDat["X"][i] > dictDat["X"][i][tgt])[0]) + 1)
        
                        total += 1
                        totalDown[dictDat["beat"][i]] += 1
                        totalPos[dictDat["pos"][i]] += 1
                        totalBeatPos[dictDat["beat"][i],dictDat["pos"][i]] += 1
                        # Accuracy:
                        if pred == tgt:
                            correct += 1
                            correctDown[dictDat["beat"][i]] += 1
                            correctPos[dictDat["pos"][i]] += 1
                            correctBeatPos[dictDat["beat"][i],dictDat["pos"][i]] += 1
                        if alphaRep != "alphaRep":
                            predMusicalDist += distMat[pred][tgt]
                        if alph != "functional":
                            predF = dictFun[reduChord(listChord[pred], alpha= 'a0', transp = 0)]
                            tgtF = dictFun[reduChord(listChord[tgt], alpha= 'a0', transp = 0)]
                            if predF == tgtF:
                                correctReducFunc +=1 
                        if alph != "functional" and alph != "a0":
                            preda0 = reduChord(listChord[pred], alpha= 'a0', transp = 0)
                            tgta0 = reduChord(listChord[tgt], alpha= 'a0', transp = 0)
                            if preda0 == tgta0:
                                correctReduca0 +=1 
                        if alph != "functional":
                            Analyzer.compare(chord = listChord[pred], target = listChord[tgt], key = listKey[dictDat["key"][i]], base_alpha = a5, print_comparison = False)
                            root_target, qual_target = parse_mir_label(listChord[tgt])
                            root_target = normalized_note(root_target)
                            root_target, qual_target = functional_tetrad(root_target, qual_target, listKey[dictDat["key"][i]], base_alpha = a5)
                            degree_target = degree(root_target, qual_target, listKey[dictDat["key"][i]])
                            if degree_target == "non-diatonic":
                                nbTotalNoDiat += 1
                                AnalyzerNoDiat.compare(chord = listChord[pred], target = listChord[tgt], key = listKey[dictDat["key"][i]], base_alpha = a5, print_comparison = False)
                                totalNoDiat += 1
                                dictKeyChordAll[listKey[dictDat["key"][i]]][tgt] += 1
                                if pred == tgt:
                                    correctNoDiat += 1
                                    # histogramm for each key on non diatonic target
                                    dictKeyChord[listKey[dictDat["key"][i]]][tgt] += 1
                                    nbCorrectChordNoDiat += 1
                            else:
                                AnalyzerDiat.compare(chord = listChord[pred], target = listChord[tgt], key = listKey[dictDat["key"][i]], base_alpha = a5, print_comparison = False)
                                totalDiat += 1
                                nbTotalDiat += 1
                                if pred == tgt:
                                    correctDiat += 1
                                    nbCorrectChordDiat += 1
                                
                        if model == "mlpDecimAug" or model == "mlpDecimAugUp":
                            if dictDat["key"][i] == dictDat["keyPred"][i]:
                                keycorrect += 1
                            if dictDat["beat"][i] == dictDat["beatPred"][i]:   
                                downbeatcorrect += 1
                    #sPrev, nby = perplexityAccumulate(dictDat["X"].tolist(), dictDat["y"].tolist(), sPrev, nby, True)
                    perp.append(perplexity(dictDat["X"].tolist(), dictDat["y"].tolist(), True))
                    totRank.append(np.mean(rank))
                    rank = []
                    acc = correct/total
                    keyacc = keycorrect/total
                    downbeatacc = downbeatcorrect/total
                    if alph != "functional":
                        accDiat = correctDiat/totalDiat
                        accNoDiat = correctNoDiat/totalNoDiat
                    accDownbeat = [int(b) / int(m) for b,m in zip(correctDown, totalDown)]
                    accPos = [int(b) / int(m) for b,m in zip(correctPos, totalPos)]
                    for i in range(len(totalBeatPos)):
                        accBeatPos[i] = [int(b) / int(m) for b,m in zip(correctBeatPos[i], totalBeatPos[i])]
                       
                    totAcc.append(acc)
                    totAccDiat.append(accDiat)
                    totAccNoDiat.append(accNoDiat)
                    totKey.append(keyacc)
                    totDownbeat.append(downbeatacc)
                    if alph != "functional":
                        accFun = correctReducFunc/total
                        totAccFun.append(accFun)
                    if alph != "functional":
                        acca0 = correctReduca0/total
                        totAcca0.append(acca0)
                    '''
                    totAcc2.append(acc2)
                    totAcc4.append(acc4)
                    totAccDownbeat.append(accDownbeat)
                    totAccPos.append(accPos)
                    totAccBeatPos.append(accBeatPos)
                    '''
                    if alph != "functional": 
                        totDist.append(musicalDist/total)
                        predTotDist.append(predMusicalDist/total)
                    
                #Pinting time !
                #perp = perplexityCompute(sPrev, nby)
                f = open("histo_output/" + model + "_" + alph + "_" + str(maxReps) + "histoChord.txt","w")
                for key, value in dictKeyChord.items():
                    f.write("Histogramme on key : " + key + "\n")
                    for nBchord in range(len(value)):
                        f.write(listChord[nBchord] + ' = ' + str(value[nBchord])+ "\n")
                    f.write("\n\n")
                f.close()
                f = open("histo_output/" + model + "_" + alph + "_" + str(maxReps) + "histoChordAll.txt","w")
                for key, value in dictKeyChordAll.items():
                    f.write("Histogramme on key : " + key + "\n")
                    for nBchord in range(len(value)):
                        f.write(listChord[nBchord] + ' = ' + str(value[nBchord])+ "\n")
                    f.write("\n\n")
                f.close()
                f = open("histo_output/" + model + "_" + alph + "_" + str(maxReps) + "histoChordRatio.txt","w")
                for key, value in dictKeyChordAll.items():
                    f.write("Histogramme on key : " + key + "\n")
                    for nBchord in range(len(value)):
                        if value[nBchord] != 0:
                            f.write(listChord[nBchord] + ' = ' + str(dictKeyChord[key][nBchord]/value[nBchord])+ "\n")
                    f.write("\n\n")
                f.close()
                #save as pickle
                f = open("histo_output/" + model + "_" + alph + "_" + str(maxReps) + "histoChord.pkl","wb")
                pickle.dump(dictKeyChord,f)
                f.close()
                #save as pickle
                f = open("histo_output/" + model + "_" + alph  + "_" + str(maxReps) + "histoChordAll.pkl","wb")
                pickle.dump(dictKeyChordAll,f)
                f.close()
                print("nbCorrectChordDiat :" + str(nbCorrectChordDiat))
                print("nbCorrectChordNoDiat :" + str(nbCorrectChordNoDiat))
                print("nbTotalDiat :" + str(nbTotalDiat))
                print("nbTotalNoDiat :" + str(nbTotalNoDiat))
                print("rank for " + model + " on alphabet " + alph + ": " + str(np.mean(totRank)))
                print("perp for " + model + " on alphabet " + alph + ": " + str(np.mean(perp)))
                print("acc for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc)))
                print("accDiat for " + model + " on alphabet " + alph + ": " + str(np.mean(totAccDiat)))
                print("accNoDiat for " + model + " on alphabet " + alph + ": " + str(np.mean(totAccNoDiat)))
                if alph != "functional":
                    print("accFun for " + model + " on alphabet " + alph + ": " + str(np.mean(totAccFun)))
                if alph != "functional" and alph != "a0":
                    print("acca0 for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcca0)))
                print("rank_std for " + model + " on alphabet " + alph + ": " + str(np.std(totRank)))
                print("perp_std for " + model + " on alphabet " + alph + ": " + str(np.std(perp)))
                print("acc_std for " + model + " on alphabet " + alph + ": " + str(np.std(totAcc)))
                print("accDiat_std for " + model + " on alphabet " + alph + ": " + str(np.std(totAccDiat)))
                print("accNoDiat_std for " + model + " on alphabet " + alph + ": " + str(np.std(totAccNoDiat)))
                if alph != "functional":
                    print("accFun_std for " + model + " on alphabet " + alph + ": " + str(np.std(totAccFun)))
                if alph != "functional" and alph != "a0":
                    print("acca0_std for " + model + " on alphabet " + alph + ": " + str(np.std(totAcca0)))
                #print("acc2 for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc2)))
                #print("acc4 for " + model + " on alphabet " + alph + ": " + str(np.mean(totAcc4)))
                print("accDownbeat for " + model + " on alphabet " + alph + ": " + str(np.average(totAccDownbeat,axis=0)))
                print("accPos for " + model + " on alphabet " + alph + ": " + str(np.average(totAccPos,axis=0)))
                print("accBeatPos for " + model + " on alphabet " + alph + ": " + str(np.average(totAccBeatPos, axis = 0)))
                if alph != "functional": 
                    print("Average Musical Distance for " + model + " on alphabet " + alph + ": " + str(np.mean(totDist)))
                    print("Average Prediction Musical Distance for " + model + " on alphabet " + alph + ": " + str(np.mean(predTotDist)))
                if model == "mlpDecimAug" or model == "mlpDecimAugUp":
                    print("accKey for " + model + " on alphabet " + alph + ": " + str(np.mean(totKey)))
                    print("accBeat for " + model + " on alphabet " + alph + ": " + str(np.mean(totDownbeat)))
                    print("accKey_std for " + model + " on alphabet " + alph + ": " + str(np.std(totKey)))
                    print("accBeat_std for " + model + " on alphabet " + alph + ": " + str(np.std(totDownbeat)))
                dictModel = {}
                dictCurrent = {}
                #basic info
                dictModel["numParam"] = dictDat["numParam"] 
                dictModel["alpha"] = dictDat["alpha"] 
                dictModel["modelType"] = dictDat["modelType"]
                dictModel["maxReps"] = maxReps
                dictModel["rank"] = np.mean(totRank)
                dictModel["perp"] = np.mean(perp)
                dictModel["acc"] = np.mean(totAcc)
                dictModel["rank_std"] = np.std(totRank)
                dictModel["perp_std"] = np.std(perp)
                dictModel["acc_std"] = np.std(totAcc)
                #diat info
                dictModel["accDiat"] = np.mean(totAccDiat)
                dictModel["accNoDiat"] = np.mean(totAccNoDiat)
                dictModel["accDiat_std"] = np.std(totAccDiat)
                dictModel["accNoDiat_std"] = np.std(totAccNoDiat)
                #reductionInfo
                if alph != "functional":
                    dictModel["accFun"] = np.mean(totAccFun)
                    dictModel["accFun_std"] = np.std(totAccFun)
                if alph != "functional" and alph != "a0":
                    dictModel["acca0"] = np.mean(totAcca0)
                    dictModel["acca0_std"] = np.std(totAcca0)
                #position info
                dictModel["accDownbeat"] = np.average(totAccDownbeat,axis=0)
                dictModel["accPos"] = np.average(totAccPos,axis=0)
                dictModel["accBeatPos"] = np.average(totAccBeatPos, axis = 0)
                #Key beat info
                if model == "mlpDecimAug" or model == "mlpDecimAugUp":
                    dictModel["accKey"] = np.mean(totKey)
                    dictModel["accBeat"] = np.mean(totDownbeat)
                    dictModel["accKey_std"] = np.std(totKey)
                    dictModel["accBeat_std"] = np.std(totDownbeat)
                
                if alph != "functional": 
                    dictModel["MusicalDist"] = np.mean(totDist)
                    dictModel["PredMusicalDist"] = np.mean(predTotDist)
                    dictACE_stats = {}
                    dictACE_degs = {}
                    for anal, anal_name in zip([Analyzer, AnalyzerDiat, AnalyzerNoDiat],['all chords', 'diatonic target', 'non-diatonic target']):
                        dictACE_stats_cur = {}
                        dictACE_degs_cur = {}
                        print("\n------\nSTATS for chord present in :" + anal_name+ "\n------")
                        StatsErrorsSubstitutions = anal.stats_errors_substitutions(stats_on_errors_only = True)
                        print("\nSTATS ERROR SUBSTITUTIONS:\n------")
                        print("Errors explained by substitutions rules: {}% of total errors\n------".format(round(anal.total_errors_explained_by_substitutions*100.0/anal.total_errors,2)))
                        print("DETAIL ERRORS EXPLAINED BY SUBSTITUTION RULES:")
                        for error_type, stat in StatsErrorsSubstitutions.items():
                            #if stat*100 &gt; 1:
                            dictACE_stats_cur[error_type] = stat
                            if stat*100 > 1:
                            		print("{}: {}%".format(error_type, round(100*stat, 2)))
                        # print(Analyzer.total_errors_degrees)
                        # print(Analyzer.total_errors_when_non_diatonic_target)
                        # print(Analyzer.total_non_diatonic_target)
                        # print(Analyzer.degrees_analysis)
                        StatsErrorsDegrees = anal.stats_errors_degrees(stats_on_errors_only = True)
                        print("\nSTATS ERROR DEGREES:\n------")
                        if anal_name != "diatonic target":
                            print("Errors when the target is not diatonic: {}% ".format(round(anal.total_errors_when_non_diatonic_target*100.0/anal.total_non_diatonic_target,2)))
                        print("Non diatonic target in {}% of the total errors".format(round(anal.total_errors_when_non_diatonic_target*100.0/anal.total_errors,2)))
                        print("When relevant: incorrect degrees (modulo inclusions): {}% of total errors\n------".format(round(anal.total_errors_degrees*100.0/anal.total_errors,2)))
                        print("DETAIL ERRORS OF DEGREES (modulo inclusions) WHEN THE TARGET IS DIATONIC:")
                        for error_type, stat  in StatsErrorsDegrees.items():
                            #if stat*100 &gt; 1:
                            dictACE_degs_cur[error_type] = stat	
                            if stat*100 > 1:
                            		print("{}: {}%".format(error_type, round(100*stat,2)))
                        dictACE_stats[anal_name] = dictACE_stats_cur
                        dictACE_degs[anal_name] = dictACE_degs_cur
                    #dictModel["MusicalDist2a"] = np.mean(totDist2a)
                    #dictModel["MusicalDist4a"] = np.mean(totDist4a)
                    #dictModel["MusicalDist2b"] = np.mean(totDist2b)
                    #dictModel["MusicalDist4b"] = np.mean(totDist4b)
                    #dictModel["MusicalDist2c"] = np.mean(totDist2c)
                    #dictModel["MusicalDist4c"] = np.mean(totDist4c)
                    
                    #dictFinalRes[model + "_" + alph] = dictModel
                    dictCurrent["res"] = dictModel
                    dictCurrent["stats"] = dictACE_stats
                    dictCurrent["degs"] = dictACE_degs
                    dictFinalRes[model + "_" + alph + "_" + str(maxReps)] = dictCurrent
                    print("\n\n")
                    
#dictFinalRes = dictFinalRes.cpu()      
sauv = open(foldName + "_DictFinaltest.pkl","wb")
pickle.dump(dictFinalRes,sauv)
sauv.close()              
print("analyses completed")
            
#%%