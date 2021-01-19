#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 19:42:21 2019

@author: carsault
"""

#%%
import argparse
import torch
from torch.utils import data
import torch.nn as nn
import os, errno
from utilities import dataImport
from utilities import chordUtil
from utilities import modelsGen
#from utilities import transformer
from utilities.chordUtil import *
#from utilities.transformer import *
from seq2seq import seq2seqModel
from seq2seq.seq2seqModel import *
from utilities import util
from utilities.util import *
from utilities import utils
from utilities.utils import *
from utilities import loss as lossPerp
from utilities import testFunc
from utilities.testFunc import *
import utilities.distance as distances
import pickle


import torch.nn.functional as F
from torch.autograd import Variable
#from ACE_Analyzer import ACEAnalyzer
#from ACE_Analyzer.ACEAnalyzer import *

#%%
"""
###################

Argument parsing

###################
"""
parser = argparse.ArgumentParser(description='Hierarchical Latent Space')
# General
parser.add_argument('--dataFolder',   type=str,   default='a0_124_123456',    help='name of the data folder')
parser.add_argument('--batch_size',      type=int,   default="500",                                help='batch size (default: 50)')
parser.add_argument('--alpha',      type=str,   default='a0',                            help='type of alphabet')
parser.add_argument('--alphaRep',      type=str,   default='normal',                            help='type of alphabet')
parser.add_argument('--lenSeq',      type=int,   default= 8,                            help='length of input sequence')
parser.add_argument('--lenPred',      type=int,   default=8,                            help='length of predicted sequence')
parser.add_argument('--decimList', nargs="+",     type=int,   default=[1],                            help='list of decimations (default: [1])')
parser.add_argument('--latent',     type=int,   default=50,                                 help='size of the latent space (default: 50)')
parser.add_argument('--hidden',     type=int,   default=500,                                 help='size of the hidden layer (default: 500)')
parser.add_argument('--modelType',      type=str,   default='mlpDecim',                            help='type of model to evaluate')
parser.add_argument('--layer',     type=int,   default=1,                                 help='number of the hidden layer - 2 (default: 1)')
parser.add_argument('--dropRatio',     type=float,   default=0.5,                                 help='drop Out ratio (default: 0.5)')
parser.add_argument('--device',     type=str,   default="cuda",                              help='set the device (cpu or cuda, default: cpu)')
parser.add_argument('--epochs',     type=int,   default=20000,                                help='number of epochs (default: 15000)')
parser.add_argument('--lr',         type=float, default=1e-4,                               help='learning rate for Adam optimizer (default: 2e-4)')
parser.add_argument('--random_state',   type=int,   default=1,    help='seed for the random train/test split')
parser.add_argument('--light',     type=util.str2bool,   default=False,                                 help='reduce the model dimension')
parser.add_argument('--rawinput',     type=util.str2bool,   default=False,                                 help='raw input')
parser.add_argument('--maxReps',      type=int,   default= 2,                            help='maxRepetition')
# RNN Learning
parser.add_argument('--teacher_forcing_ratio',     type=float,   default=0,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--professor_forcing',     type=util.str2bool,   default=False,                                 help='activate professor forcing GAN training (default: False)')
parser.add_argument('--professor_forcing_ratio',     type=float,   default=0.0,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--attention',     type=util.str2bool,   default=True,                                 help='attention mechanism in LSTM decoder')
parser.add_argument('--expand',     type=util.str2bool,   default=True,                                 help='reduce the latent space in LSTM')
# Save file
parser.add_argument('--foldName',      type=str,   default='modelSave200908',                            help='name of the folder containing the models')
parser.add_argument('--modelName',      type=str,   default='bqwlbq',                            help='name of model to evaluate')
parser.add_argument('--dist',      type=str,   default='euclidian',                            help='distance to compare predicted sequence (default : euclidian')
args = parser.parse_args()
print(args)


#Analyzer = ACEAnalyzer()

if args.decimList[0]==1:
    if args.alphaRep == "alphaRep":
        args.dataFolder = args.alpha + "_1_" + str(args.random_state) + "newRep" + str(args.maxReps)
    else:
        args.dataFolder = args.alpha + "_124_" + str(args.random_state)
else:
    if args.alphaRep == "alphaRep":
        args.dataFolder = args.alpha + "_1_" + str(args.random_state) + "newRep" + str(args.maxReps)
    else:
        args.dataFolder = args.alpha + "_124_" + str(args.random_state)

str1 = ''.join(str(e) for e in args.decimList)

if args.modelType == "mlpDecimFamily":
    args.modelName = args.dataFolder + "_124_" + args.modelType
else:
    args.modelName = args.dataFolder + "_" + str1 + "_" + args.modelType

# Create save folder
try:
    os.mkdir("testVector")
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir("testVector/" + args.foldName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir("testVector/" + args.foldName + '/' + args.modelName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# CUDA for PyTorch
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor
'''
if args.device is not torch.device("cpu"):
    print(args.device)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark=True
'''
#%% Dataset 
if args.modelType == "mlpDecim":
    args.batch_size = 500
# Create generators
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 0}

# Create dataset
dataset_train = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/train.pkl")
dataset_valid = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/valid.pkl")
dataset_test = dataImport.createDatasetFull("datasets/" + args.dataFolder + "/test.pkl")

bornInf = {}
bornSup = {}
listNameModel = {}
#listNameModel[2] = "mlp2Decim2bis"
#listNameModel[4] = "mlpDecim4bis"
listNameModel[2] = args.dataFolder + "_2_" + "mlpDecim"
listNameModel[4] = args.dataFolder + "_4_" + "mlpDecim"
res = {}

initial_teacher_forcing_ratio = 0

bornInf[1] = 0
bornSup[1] = 8
bornInf[2] = 8
bornSup[2] = 12
bornInf[4] = 12
bornSup[4] = 14 


    
training_generator = data.DataLoader(dataset_train, pin_memory = True, **params)
validating_generator = data.DataLoader(dataset_valid, pin_memory = True, **params)
testing_generator = data.DataLoader(dataset_test, pin_memory = True, **params)

#%%

# Get chord alphabet
if args.alpha == "functional":
    dictChord = relatif
    listChord = relatifList
    n_categories = 13
    args.n_categories = n_categories
else:
    dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))
    n_categories = len(listChord)
    args.n_categories = n_categories
    
if args.alphaRep == "alphaRep":
    dictChord, listChord = chordUtil.getDictChordUpgrade(eval(args.alpha), args.maxReps)
    #n_categories = len(listChord) + 1
    n_categories = len(listChord)

def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

# Get chord distance Matrix
#dictChordDist, listChordDist = chordUtil.g%tDictChord(eval(args.alpha))
#tf_mappingR = distances.tonnetz_matrix((invert_dict(dictChordDist),invert_dict(dictChordDist)))
tf_mappingR = distances.tonnetz_matrix((invert_dict(dictChord),invert_dict(dictChord)))
tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
tf_mappingR = 1./ tf_mappingR
tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
tf_mappingR = (tf_mappingR) / np.sum(tf_mappingR[0])
tf_mappingR = Variable(torch.from_numpy(np.asarray(tf_mappingR)))

if args.alpha == "functional":
    tf_mappingR = torch.eye(13)

if args.device is not torch.device("cpu"):
    tf_mappingR = tf_mappingR.to(args.device)


dictKey, listKey = chordUtil.getDictKey()
n_categoriesInput = n_categories

print(n_categories)

decim = args.decimList[0]
if args.dist != 'None' and args.alpha != "functional":
	distMat = testFunc.computeMat(dictChordDist, args.dist)
	distMat = torch.Tensor(distMat).to(args.device,non_blocking=True)

if args.modelType == "mlpDecim":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    net = modelsGen.InOutModel(enc,dec)
    
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        #criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()

elif args.modelType == "mlpDecimFamily":
    net = modelsGen.ModelFamily() 
    for i in args.decimList:
        enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, i, args.layer, args.dropRatio)
        if i != 1 :
            dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.latent, i, args.layer, args.dropRatio)
            model = modelsGen.InOutModel(enc,dec)
            #file_name = "2019-03-11" + "mlpDecim[" + str(i) + "]seed" + str(args.random_state)
            file_name = listNameModel[i]
            model.load_state_dict(torch.load(args.foldName + '/' + str(file_name) + '/' + str(file_name) ,map_location = args.device))
            net.addModel(model, str(i))
        else:
            dec = modelsGen.DecoderFinal(args.lenSeq, args.lenPred, n_categories, args.hidden, args.latent * len(args.decimList), args.layer, args.dropRatio)
            model = modelsGen.FinalModel(enc,dec)
            net.addModel(model, str(i))
            encoder_optimizer = torch.optim.Adam(net.models[str(1)].encoder.parameters(), lr=args.lr)
            decoder_optimizer = torch.optim.Adam(net.models[str(1)].decoder.parameters(), lr=args.lr)
            enc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=0.5, threshold=1e-6)
            dec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=0.5, threshold=1e-6)
    criterion = nn.CrossEntropyLoss()
        
elif args.modelType == "mlpDecimBeat":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encTensor1 = modelsGen.EncoderMLP(args.lenSeq, 4, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    #encTensor = modelsGen.NetConv()
    decDouble = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, int(args.latent + args.latent/10), decim, args.layer, args.dropRatio)
    net = modelsGen.InOutModelDoubleBeat(enc, encTensor1, decDouble)
    criterion = nn.BCELoss()
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        #criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        
elif args.modelType == "mlpDecimKey":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encTensor2 = modelsGen.EncoderMLP(args.lenSeq, 25, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    #encTensor = modelsGen.NetConv()
    decDouble = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, int(args.latent + args.latent/10), decim, args.layer, args.dropRatio)
    net = modelsGen.InOutModelDoubleKey(enc, encTensor2, decDouble)
    criterion = nn.BCELoss()
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        #criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()

elif args.modelType == "mlpDecimKeyBeat":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encTensor1 = modelsGen.EncoderMLP(args.lenSeq, 4, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    encTensor2 = modelsGen.EncoderMLP(args.lenSeq, 25, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    #encTensor = modelsGen.NetConv()
    decDouble = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, int(args.latent + 2*args.latent/10), decim, args.layer, args.dropRatio)
    net = modelsGen.InOutModelTripleKeyBeat(enc, encTensor1, encTensor2, decDouble)
    criterion = nn.BCELoss()
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        #criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        
elif args.modelType == "mlpDecimAug":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encBeat = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, 4, decim, args.layer, args.dropRatio)
    encKey = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, 25, decim, args.layer, args.dropRatio)
    #enc = modelsGen.dilatConvBatch(args.lenSeq, 1, n_categories, args.latent)
    #enc = modelsGen.ConvNet(args)
    
    if args.rawinput:
        dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.lenSeq*args.n_categories + 4 + 25, decim, args.layer, args.dropRatio)
        net = modelsGen.InOutModelTripleRawData(encBeat, encKey, dec, args)
    else:
        dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.latent + 4 + 25, decim, args.layer, args.dropRatio)
        net = modelsGen.InOutModelTriple(enc, encBeat, encKey, dec)
    args.key = True
    args.rec = False
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        #criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
        #criterionKey = nn.BCELoss()
        criterionKey = nn.CrossEntropyLoss()
        #criterionBeat = nn.BCELoss()
        criterionBeat = nn.CrossEntropyLoss()
        
elif args.modelType == "mlpDecimAugUp":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encTensor = modelsGen.EncoderMLP(args.lenSeq, args.lenSeq, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    decDouble = modelsGen.DecoderMLPKey(1, args.latent, args.hidden, int(args.latent + args.latent/10), decim, args.layer - 1, args.dropRatio)
    netRec = modelsGen.InOutModelDouble(enc, encTensor, decDouble)
    args.key = True
    args.rec = False
    if args.light:
        netBeat = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, 4, decim, args.layer - 1, args.dropRatio)
        netKey = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, 25, decim, args.layer - 1, args.dropRatio)
    else:
        encBeat = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, args.latent, decim, args.layer, args.dropRatio)
        encTensorBeat = modelsGen.EncoderMLP(args.lenSeq, args.lenSeq, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
        decDoubleBeat = modelsGen.DecoderMLPKey(1, 4, args.hidden, int(args.latent + args.latent/10), decim, args.layer - 1, args.dropRatio)
        netBeat = modelsGen.InOutModelDouble(encBeat, encTensorBeat, decDoubleBeat)
        
        encKey = modelsGen.EncoderMLP(args.lenSeq, n_categoriesInput, args.hidden, args.latent, decim, args.layer, args.dropRatio)
        encTensorKey = modelsGen.EncoderMLP(args.lenSeq, args.lenSeq, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
        decDoubleKey = modelsGen.DecoderMLPKey(1, 25, args.hidden, int(args.latent + args.latent/10), decim, args.layer - 1, args.dropRatio)
        netKey = modelsGen.InOutModelDouble(encKey, encTensorKey, decDoubleKey)
    
    #enc = modelsGen.dilatConvBatch(args.lenSeq, 1, n_categories, args.latent)
    #enc = modelsGen.ConvNet(args)
    dec = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, args.latent + 4 + 25, decim, args.layer, args.dropRatio)
    if args.light:
        net = modelsGen.InOutModelTripleMatrixBis(netRec, netBeat, netKey, dec)
    else:
        net = modelsGen.InOutModelTripleMatrix(netRec, netBeat, netKey, dec)
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
        criterionKey = nn.BCELoss()
        criterionBeat = nn.BCELoss()
        
elif args.modelType == "mlpDecimTensor":
    enc = modelsGen.EncoderMLP(args.lenSeq, n_categories, args.hidden, args.latent, decim, args.layer, args.dropRatio)
    encTensor = modelsGen.EncoderMLP(args.lenSeq, args.lenSeq, int(args.hidden/50), int(args.latent/10), decim, args.layer, 0)
    #enc = modelsGen.dilatConv(args.lenSeq, 1, n_categories, args.latent)
    decDouble = modelsGen.DecoderMLP(args.lenPred, n_categories, args.hidden, int(args.latent + args.latent/10), decim, args.layer, args.dropRatio)
    if args.rawinput:
        net = modelsGen.InOutModelDoubleRawData(encTensor, decDouble, args)
    else:
        net = modelsGen.InOutModelDouble(enc, encTensor, decDouble)
    criterion = nn.BCELoss()
    if args.decimList[0] != 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

elif args.modelType == "lstmDecim":
    if args.attention == True:
        enc = seq2seqModel.EncoderRNNattention(args, n_categories + 2, args.hidden, args.latent, args.layer + 1 , expand = args.expand)
        dec = seq2seqModel.DecoderRNNattention(args, n_categories + 2, args.hidden, args.latent, args.layer + 1, attention = args.attention, expand = args.expand)
    else:
        enc = seq2seqModel.EncoderRNN(args, n_categories + 2, args.hidden, args.layer + 1)
        dec = seq2seqModel.DecoderRNN(args, n_categories + 2, args.hidden, args.layer + 1)
    encoder_optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=1e-4)
    decoder_optimizer = torch.optim.Adam(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    #encoder_optimizer = torch.optim.SGD(enc.parameters(), lr=args.lr)
    #decoder_optimizer = torch.optim.SGD(dec.parameters(), lr=args.lr)
    #schedulerEnc = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size = 20 ,gamma = 0.1)
    enc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=0.5, threshold=1e-6)
    #schedulerDec = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size = 20 ,gamma = 0.1)
    dec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=0.5, threshold=1e-6)
    net = seq2seqModel.Seq2Seq(enc, dec, args, encoder_optimizer, decoder_optimizer)
    #criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    
    
    discriminator = seq2seqModel.Discriminator(args.layer + 1, args.hidden)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    criterionDicrim = nn.CrossEntropyLoss()
    if args.professor_forcing == True:
        net = seq2seqModel.Seq2Seq(enc, dec, args, encoder_optimizer, decoder_optimizer, discriminator, discriminator_optimizer)
else:
    print("Bad model type")
# Print model 
print(net)
#f.write(print(net))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(net))
res['numberOfModelParams'] = count_parameters(net)

if args.device is not "cpu":
    net.to(args.device)

#if   args.modelType != "transformer":
# Choose lose
#if args.decimList[0] != 1:
    #criterion = nn.MSELoss()
    #else:
    #    criterion = nn.BCELoss()
    
# choose optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()



accDiscr = 1
bestAccurValid = 0
accurValid = 0
bestValid_total_loss = 1000
test_total_loss = 0

#perp = lossPerp.Perplexity()
# Begin testing
net.load_state_dict(torch.load(args.foldName + '/' + args.modelName + '/' + args.modelName, map_location = args.device))
print(net)
#%%
listX = []
listy = []
listkey = []
listbeat = []
listkeyPred = []
listbeatPred = []
listpos = []
dictDat = {}
for local_batch, local_labels, local_key, local_beat in testing_generator:
    if args.alphaRep == "alphaRep":
        local_batch = nn.functional.one_hot(local_batch.long(),n_categories)           
        local_labels = nn.functional.one_hot(local_labels.long(),n_categories)
    torch.cuda.empty_cache()
    if len(args.decimList) == 1:
        local_batch = local_batch[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
    local_labels = local_labels[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous() 
    local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
    local_batchRep = local_batch
    #local_batch = local_batch.view(len(local_batch),1,8,25)
    local_beat = local_beat.to(args.device,non_blocking=True)
    local_key = local_key.to(args.device,non_blocking=True)
    if args.modelType == "mlpDecim":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            if args.alphaRep == "alphaRep":
                output = net(local_batch.float())
            else:
                output = net(local_batch)
            if args.decimList[0] != 1:
                loss = criterion(output, local_labels)
            else:
                output = output.transpose(1,2)
                topv, topi = local_labels.topk(1)
                topi = topi[:,:,0]
                #loss = criterion(output, local_labels)
                loss = criterion(output, topi)
                output = output.transpose(1,2)
            test_total_loss += loss
            
    if args.modelType == "mlpDecimKey":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            local_k = keyToOneHot(local_key)
            output = net(local_batch, local_k)
            output = output.transpose(1,2)
            topv, topi = local_labels.topk(1)
            topi = topi[:,:,0]
            #loss = criterion(output, local_labels)
            loss = criterion(output, topi)
            output = output.transpose(1,2)
            test_total_loss += loss
            
    if args.modelType == "mlpDecimBeat":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            local_b = beatToOneHot(local_beat)
            output = net(local_batch, local_b)
            output = output.transpose(1,2)
            topv, topi = local_labels.topk(1)
            topi = topi[:,:,0]
            #loss = criterion(output, local_labels)
            loss = criterion(output, topi)
            output = output.transpose(1,2)
            test_total_loss += loss
            
    if args.modelType == "mlpDecimKeyBeat":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            local_k = keyToOneHot(local_key)
            local_b = beatToOneHot(local_beat)
            output = net(local_batch, local_k, local_b)
            output = output.transpose(1,2)
            topv, topi = local_labels.topk(1)
            topi = topi[:,:,0]
            #loss = criterion(output, local_labels)
            loss = criterion(output, topi)
            output = output.transpose(1,2)
            test_total_loss += loss
            
    if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            if args.modelType == "mlpDecimAug":
                #output, beat, key = net(local_batch)
                if args.alphaRep == "alphaRep":
                    output, beat, key = net(local_batch.float())
                else:
                    output, beat, key = net(local_batch)
            else:
                tensorSim = computeTensor(local_batch, tf_mappingR.float())
                output, beat, key = net(local_batch, tensorSim)
            output = output.transpose(1,2)
            topv, topi = local_labels.topk(1)
            topi = topi[:,:,0]
            #loss = criterion(output, local_labels)
            loss = criterion(output, topi)
            output = output.transpose(1,2)
            #loss = criterion(output, local_labels)
            test_total_loss += loss
            
    if args.modelType == "mlpDecimTensor":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            tensorSim = computeTensor(local_batch, tf_mappingR.float())
            output = net(local_batch, tensorSim)
            loss = criterion(output, local_labels)
            test_total_loss += loss

    if args.modelType == "mlpDecimFamily":
        with torch.no_grad():
            net.eval() 
            net.zero_grad()
            output = net(local_batch, args, bornInf, bornSup)
            output = output.transpose(1,2)
            topv, topi = local_labels.topk(1)
            topi = topi[:,:,0]
            #loss = criterion(output, local_labels)
            loss = criterion(output, topi)
            output = output.transpose(1,2)
            #loss = criterion(output, local_labels)
            test_total_loss += loss

    if args.modelType == "lstmDecim":
        with torch.no_grad():
            if args.alphaRep == "alphaRep":
                decoder_output, accDiscr = net.test(local_batch.float())
            else:
                decoder_output, accDiscr = net.test(local_batch)
            output = decoder_output[:,0,:,:].transpose(0,1).view(len(local_batch),int(args.lenPred/decim),n_categories+2)
            

    if args.decimList[0] == 1:
    
        for i in range(output.size()[0]):
            if args.alphaRep == "alphaRep":
                output_bis, local_labels_bis = transfToPrevRep(args,output[i],local_labels[i])
                output_bis = nn.functional.one_hot(torch.Tensor(output_bis).long(),args.n_categories)           
                local_labels_bis = nn.functional.one_hot(torch.Tensor(local_labels_bis).long(),args.n_categories)
                listX.append(output_bis)
                listy.append(local_labels_bis)
            else:
                listX.append(output[i])
                listy.append(local_labels[i])
            for j in range(int(args.lenPred/decim)):
                #listX.append(output[i][j])
                #listy.append(local_labels[i][j])
                listkey.append(int(local_key[i][j].item()))
                listbeat.append(int(local_beat[i][j].item()))
                listpos.append(j)
                if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
                    listkeyPred.append(key[i].data.max(0, keepdim=False)[1])
                    listbeatPred.append(beat[i].data.max(0, keepdim=False)[1]+j)
                
listX = torch.cat(listX, 0)
listy = torch.cat(listy, 0)
dictDat["X"] = listX
dictDat["y"] = listy
dictDat["key"] = listkey
dictDat["beat"] = listbeat
dictDat["pos"] = listpos
dictDat["keyPred"] = listkeyPred
dictDat["beatPred"] = listbeatPred
dictDat["numParam"] = res['numberOfModelParams']
dictDat["alpha"] = args.alpha
dictDat["modelType"] = args.modelType

dictDat["random_state"] = args.random_state
dictDat["modelName"] = args.modelName

sauv = open("testVector/" + args.foldName + '/' + args.modelName + '/' + "probVect_" + args.modelName + "_test.pkl","wb")
pickle.dump(dictDat,sauv)
sauv.close()
print("done!")
            
