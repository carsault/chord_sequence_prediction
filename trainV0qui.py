#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:14:34 2020

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
import time

#time.sleep(3600)

import torch.nn.functional as F
from torch.autograd import Variable
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
parser.add_argument('--n_categories',     type=int,   default=0,                                 help='n of categories')
parser.add_argument('--latent',     type=int,   default=50,                                 help='size of the latent space (default: 50)')
parser.add_argument('--hidden',     type=int,   default=500,                                 help='size of the hidden layer (default: 500)')
parser.add_argument('--modelType',      type=str,   default='mlpDecim',                            help='type of model to evaluate')
parser.add_argument('--layer',     type=int,   default=1,                                 help='number of the hidden layer - 2 (default: 1)')
parser.add_argument('--dropRatio',     type=float,   default=0.4,                                 help='drop Out ratio (default: 0.5)')
parser.add_argument('--device',     type=str,   default="cuda",                              help='set the device (cpu or cuda, default: cpu)')
parser.add_argument('--epochs',     type=int,   default=500,                                help='number of epochs (default: 15000)')
parser.add_argument('--lr',         type=float, default=1e-3,                               help='learning rate for Adam optimizer (default: 2e-4)')
parser.add_argument('--random_state',   type=int,   default=1,    help='seed for the random train/test split')
parser.add_argument('--warmup',   type=int,   default=0,    help='number of epochs for warmup')
parser.add_argument('--early',   type=int,   default=30,    help='number of epochs before earlystopping')
parser.add_argument('--light',     type=util.str2bool,   default=False,                                 help='reduce the model dimension')
parser.add_argument('--key',     type=util.str2bool,   default=False,                                 help='train on key')
parser.add_argument('--beat',     type=util.str2bool,   default=False,                                 help='train on beat')
parser.add_argument('--rec',     type=util.str2bool,   default=True,                                 help='train rec')
parser.add_argument('--rawinput',     type=util.str2bool,   default=False,                                 help='raw input')
parser.add_argument('--maxReps',      type=int,   default= 2,                            help='maxRepetition')
# RNN Learning
parser.add_argument('--teacher_forcing_ratio',     type=float,   default=0.5,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--teacher_forcing',     type=util.str2bool,   default=True,                                 help='activate teacher forcing')
parser.add_argument('--professor_forcing',     type=util.str2bool,   default=False,                                 help='activate professor forcing GAN training (default: False)')
parser.add_argument('--professor_forcing_ratio',     type=float,   default=0,                                 help='between 0 and 1 (default: 0.5)')
parser.add_argument('--attention',     type=util.str2bool,   default=True,                                 help='attention mechanism in LSTM decoder')
parser.add_argument('--expand',     type=util.str2bool,   default=False,                                 help='reduce the latent space in LSTM')
# Save file
parser.add_argument('--foldName',      type=str,   default='modelSave190515',                            help='name of the folder containing the models')
parser.add_argument('--modelName',      type=str,   default='bqwlbq',                            help='name of model to evaluate')
parser.add_argument('--specName',      type=str,   default='',                            help='name of model to evaluate')
parser.add_argument('--dist',      type=str,   default='euclidian',                            help='distance to compare predicted sequence (default : euclidian')
args = parser.parse_args()
print(args)

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

print(args.dataFolder)
args.modelName = args.dataFolder + "_" + str1 + "_" + args.modelType + args.specName

# Create save folder
try:
    os.mkdir(args.foldName)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir(args.foldName + '/' + args.modelName)
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
initial_teacher_forcing_ratio = args.teacher_forcing_ratio

listNameModel[2] = args.dataFolder + "_2_" + "mlpDecim"
listNameModel[4] = args.dataFolder + "_4_" + "mlpDecim"
res = {}

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
else:
    dictChord, listChord = chordUtil.getDictChord(eval(args.alpha))
    n_categories = len(dictChord)
    
if args.alphaRep == "alphaRep":
    dictChord, listChord = chordUtil.getDictChordUpgrade(eval(args.alpha), args.maxReps)
    #n_categories = len(dictChord)+1
    n_categories = len(dictChord)

def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

# Get chord distance Matrix
'''
tf_mappingR = distances.tonnetz_matrix((invert_dict(dictChord),invert_dict(dictChord)))
tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
tf_mappingR = 1./ tf_mappingR
tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
tf_mappingR = (tf_mappingR) / np.sum(tf_mappingR[0])
tf_mappingR = Variable(torch.from_numpy(np.asarray(tf_mappingR)))

if args.alpha == "functional":
    tf_mappingR = torch.eye(13)
'''
tf_mappingR = torch.eye(len(listChord))

if args.device is not torch.device("cpu"):
    tf_mappingR = tf_mappingR.to(args.device)

#%%
# Model definition
#n_categories = len(listChord)
print(dictChord)
#n_categories = len(tf_mappingR[0])
n_categoriesInput = n_categories
args.n_categories = n_categories
print(n_categories)
#def cross_entropy_one_hot(input, target):
#    _, labels = target.max(dim=0)
#    return nn.CrossEntropyLoss()(input, labels)

decim = args.decimList[0]
'''
if args.dist != 'None':
	distMat = testFunc.computeMat(dictChord, args.dist)
	distMat = torch.Tensor(distMat).to(args.device,non_blocking=True)
'''

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

if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
    if args.rawinput:
        optimizerRecEnc = None
    else:
        optimizerRecEnc = torch.optim.Adam(net.encoder.parameters(),lr=args.lr)
        enc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerRecEnc, factor=0.5, threshold=1e-6)
    optimizerRecDec = torch.optim.Adam(net.decoder.parameters(),lr=args.lr)
    dec_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerRecDec, factor=0.5, threshold=1e-6)
    optimizerBeat = torch.optim.Adam(net.encoderTensor1.parameters(),lr=args.lr)
    optimizerKey = torch.optim.Adam(net.encoderTensor2.parameters(),lr=args.lr)

else:
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20 ,gamma = 0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-6)


accDiscr = 1
bestAccurValid = 0
accurValid = 0
bestValid_total_loss = 1000
earlyStop = 0

ealyStopKey = 10
ealyStopBeat = 10
bestAccurKey = 0
bestAccurBeat = 0
accurBeat = 0
accurKey = 0

epochDecay = 0

#perp = lossPerp.Perplexity()
# Begin training
bestValid_total_loss = 10000000
for epoch in range(args.epochs):
    print('Epoch number {} '.format(epoch))
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    # Training
    train_total_loss = 0
    train_keytotal_loss = 0
    train_beattotal_loss = 0
    
    valid_total_loss = 0
    test_total_loss = 0
    
    
    if args.modelType == "mlpDecim" or args.modelType == "mlpDecimKey" or args.modelType == "mlpDecimBeat" or args.modelType == "mlpDecimKeyBeat":
        train_total_loss = net.train_epoch(training_generator, optimizer, criterion, bornInf, bornSup, tf_mappingR, args)  
    if args.modelType == "mlpDecimFamily":
        train_total_loss = net.train_epoch(training_generator, encoder_optimizer, decoder_optimizer, criterion, bornInf, bornSup, tf_mappingR, args)
    if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
        train_total_loss, train_keytotal_loss, train_beattotal_loss = net.train_epoch(training_generator, optimizerRecEnc, optimizerRecDec, optimizerKey, optimizerBeat, criterion, criterionKey, criterionBeat, bornInf, bornSup, tf_mappingR, args)
    if args.modelType == "lstmDecim":
        train_total_loss, train_total_lossD = net.train_epoch(training_generator, criterion, criterionDicrim, accDiscr, bornInf, bornSup, epoch, args)
            
    print("loss rec" + str(train_total_loss))
    print("loss key" + str(train_keytotal_loss))
    print("loss beat" + str(train_beattotal_loss))
    
    # Validation
    totalVal = 0
    correct = 0
    totalKey = 0
    correctKey = 0
    totalBeat = 0
    correctBeat = 0
    
    for local_batch, local_labels, local_key, local_beat in validating_generator:
        if args.alphaRep == "alphaRep":
            local_batch = nn.functional.one_hot(local_batch.long(),n_categories)           
            local_labels = nn.functional.one_hot(local_labels.long(),n_categories)
        if len(args.decimList) == 1:
            local_batch = local_batch[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
        local_labels = local_labels[:,bornInf[args.decimList[0]]:bornSup[args.decimList[0]],:].contiguous()
        local_batch, local_labels = local_batch.to(args.device,non_blocking=True), local_labels.to(args.device,non_blocking=True)
        #local_batch = local_batch.view(len(local_batch),1,8,25)
        #local_batch = someTorchMusicalRandomTransform(local_batch) -> for example by using the tf_mappingR prob to exchange some chords
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
                valid_total_loss += loss
                
        if args.modelType == "mlpDecimKey":
            with torch.no_grad():
                net.eval() 
                net.zero_grad()
                local_key = keyToOneHot(local_key)
                output = net(local_batch, local_key)
                output = output.transpose(1,2)
                topv, topi = local_labels.topk(1)
                topi = topi[:,:,0]
                #loss = criterion(output, local_labels)
                loss = criterion(output, topi)
                output = output.transpose(1,2)
                valid_total_loss += loss
                
        if args.modelType == "mlpDecimBeat":
            with torch.no_grad():
                net.eval() 
                net.zero_grad()
                local_beat = beatToOneHot(local_beat)
                output = net(local_batch, local_beat)
                output = output.transpose(1,2)
                topv, topi = local_labels.topk(1)
                topi = topi[:,:,0]
                #loss = criterion(output, local_labels)
                loss = criterion(output, topi)
                output = output.transpose(1,2)
                valid_total_loss += loss
                
        if args.modelType == "mlpDecimKeyBeat":
            with torch.no_grad():
                net.eval() 
                net.zero_grad()
                local_key = keyToOneHot(local_key)
                local_beat = beatToOneHot(local_beat)
                output = net(local_batch, local_key, local_beat)
                output = output.transpose(1,2)
                topv, topi = local_labels.topk(1)
                topi = topi[:,:,0]
                #loss = criterion(output, local_labels)
                loss = criterion(output, topi)
                output = output.transpose(1,2)
                valid_total_loss += loss
                
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
                #print(output.size())
                #print(topi.size())
                loss = criterion(output, topi)
                output = output.transpose(1,2)
                #loss = criterion(output, local_labels)
                valid_total_loss += loss

        if args.modelType == "mlpDecimTensor":
            with torch.no_grad():
                net.eval() 
                net.zero_grad()
                tensorSim = computeTensor(local_batch, tf_mappingR.float())
                output = net(local_batch, tensorSim)
                loss = criterion(output, local_labels)
                valid_total_loss += loss

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
                valid_total_loss += loss
                
        if args.modelType == "lstmDecim":
            with torch.no_grad():
                if args.alphaRep == "alphaRep":
                    decoder_output, accDiscr = net.test(local_batch.float())
                else:
                    decoder_output, accDiscr = net.test(local_batch)
                output = decoder_output[:,0,:,:].transpose(0,1).view(len(local_batch),int(args.lenPred/decim),n_categories+2)


        lablbatmax = local_labels.data.max(2, keepdim=False)[1]
        totalVal += output.size()[0] * output.size()[1]
        pred = output.data.max(2, keepdim=False)[1]
        
        if args.alphaRep == "alphaRep":
            batch_correct,_ = chordUtil.computeAccNewRep(args,decim,output,local_labels)
            correct += batch_correct
        else:
            correct += pred.eq(lablbatmax.data.view_as(pred)).sum().item()
           
        if args.key == True:
            #local_key = local_key[:,0,:]
            #keybatmax = local_key.data.max(1, keepdim=False)[1]
            keybatmax = local_key[:,0].long()
            totalKey += output.size()[0]
            pred = key.data.max(1, keepdim=False)[1]
            correctKey += pred.eq(keybatmax.data.view_as(pred)).sum().item()
            accurKey = 100 * correctKey / totalKey
            
        if args.beat == True:
            #local_beat = local_beat[:,0,:]
            #beatbatmax = local_beat.data.max(1, keepdim=False)[1]
            beatbatmax = local_beat[:,0].long()
            totalBeat += output.size()[0]
            pred = beat.data.max(1, keepdim=False)[1]
            correctBeat += pred.eq(beatbatmax.data.view_as(pred)).sum().item()
            accurBeat = 100 * correctBeat / totalBeat

    accurValid = 100 * correct / totalVal
    

    if args.decimList[0] == 1:
        if args.modelType == "mlpDecimFamily" or args.modelType == "lstmDecim" or args.modelType == "mlpDecimAug":
            enc_lr_scheduler.step(100-accurValid)
            dec_lr_scheduler.step(100-accurValid)
        else:
            lr_scheduler.step(100-accurValid)
        if  accurValid > bestAccurValid and args.rec:
            bestAccurValid = accurValid
            res["params"] = str(args)
            res["bestAccurValid"] = bestAccurValid
            res["epochOnBestAccurValid"] = epoch
            print("new best loss, model saved")
            print('New accuracy of the network on valid dataset: {} %'.format(bestAccurValid))
            torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)
            earlyStop = 0
    
        else:
            if epoch > args.warmup and args.rec :
                earlyStop += 1
                print("increasing early stopping")
                print('Old best accuracy of the network on valid dataset: {} %'.format(bestAccurValid))
                print('current accuracy of the network on valid dataset: {} %'.format(accurValid))
                #args.lr = args.lr * 0.9
                #if args.modelType == "lstmDecim":
                #    schedulerEnc.step()
                #    schedulerDec.step()
                #print("new lr =" + str(args.lr))
                #print("decreasing lr to " + str(args.lr))
                
    if args.decimList[0] != 1:
        lr_scheduler.step(valid_total_loss)
        if  valid_total_loss < bestValid_total_loss and args.rec:
            bestValid_total_loss = valid_total_loss
            res["params"] = str(args)
            res["bestValidLoss"] =bestValid_total_loss
            res["epochOnBestAccurValid"] = epoch
            res["epochOnBestPrep"] = epoch
            print("new best loss, model saved")
            torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)
            earlyStop = 0
    
        else:
            if epoch > args.warmup and args.rec :
                earlyStop += 1
                print("increasing early stopping")
                print('Old best loss of the network on valid dataset: {} %'.format(bestValid_total_loss))
                print('current lost of the network on valid dataset: {} %'.format(valid_total_loss))
                #args.lr = args.lr * 0.9
                #if args.modelType == "lstmDecim":
                #    schedulerEnc.step()
                #    schedulerDec.step()
                #print("new lr =" + str(args.lr))
                #print("decreasing lr to " + str(args.lr))              
                
            
    if epoch > args.warmup and args.rec :

        if args.teacher_forcing == True:
            tfr = initial_teacher_forcing_ratio * (30-epoch)/30
            #tfr = 1 - epochDecay/args.epochs
            args.teacher_forcing_ratio = tfr if tfr > 0 else 0
            print("decreasing teacher forcing to " + str(args.teacher_forcing_ratio))
            if args.teacher_forcing_ratio != 0 and args.modelType == "lstmDecim":
                earlyStop = 0
        
        '''
        epochDecay += 1 
        if args.decimList[0] == 1:
            lr = args.lr * (0.1 ** (epochDecay // 20))
        #else:
        #    lr = args.lr * (0.1 ** (epochDecay // 50))
        
        print("decreasing lr to " + str(lr))    
        if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
            if args.rawinput:
                optimizerRecEnc = None
            else:
                optimizerRecEnc = torch.optim.Adam(net.encoder.parameters(),lr=lr)
            optimizerRecDec = torch.optim.Adam(net.decoder.parameters(),lr=lr)
            optimizerBeat = torch.optim.Adam(net.encoderTensor1.parameters(),lr=lr)
            optimizerKey = torch.optim.Adam(net.encoderTensor2.parameters(),lr=lr)
        
        elif args.modelType == "lstmDecim":
            #encoder_optimizer = torch.optim.Adam(enc.parameters(), lr=lr)
            #decoder_optimizer = torch.optim.Adam(dec.parameters(), lr=lr)
            schedulerEnc.step()
            schedulerDec.step()
        else:
            optimizer = torch.optim.Adam(net.parameters(),lr=lr)
            
            if args.modelType == "mlpDecimFamily":
                encoder_optimizer = torch.optim.Adam(net.parameters(),lr=lr)
                decoder_optimizer = torch.optim.Adam(net.parameters(),lr=lr)
            scheduler.step()
        
     '''
    #Only for MLP Aug
    if accurKey > bestAccurKey and args.key:
        ealyStopKey = 10
        bestAccurKey = accurKey
        res["bestAccurValidKey"] = bestAccurKey
        res["epochOnBestAccurValidKey"] = epoch
        print('New accuracy of Key on valid dataset: {} %'.format(bestAccurKey))
        torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)        
    else:
        ealyStopKey -= 1
        if ealyStopKey == 0 and args.key:
            args.key = False
            args.beat = True
            net.load_state_dict(torch.load(args.foldName + '/' + args.modelName + '/' + args.modelName, map_location = args.device))
            print('Stop Key training with acc on valid dataset: {} %'.format(bestAccurKey))
            
    if accurBeat > bestAccurBeat and args.beat:
        bestAccurBeat = accurBeat
        ealyStopBeat = 10
        res["bestAccurValidBeat"] = bestAccurBeat
        res["epochOnBestAccurValidBeat"] = epoch
        print('New accuracy of Beat on valid dataset: {} %'.format(bestAccurBeat))
        torch.save(net.state_dict(), args.foldName + '/' + args.modelName + '/' + args.modelName)        
    else:
        if args.beat:
            ealyStopBeat -= 1
        if ealyStopBeat == 0 and args.beat:
            args.beat = False
            args.rec = True
            net.load_state_dict(torch.load(args.foldName + '/' + args.modelName + '/' + args.modelName, map_location = args.device))
            print('Stop Beat training with acc on valid dataset: {} %'.format(bestAccurBeat))
            
    totalTest = 0
    correct = 0
    correctrepeat = 0

    accuraList = [0] * int(args.lenPred/decim)
    musicalD = 0
    accurTest = 0
    accurRepeat = 0
    if earlyStop > args.early or epoch > args.epochs - 2:
        totalKey = 0
        correctKey = 0
        totalBeat = 0
        correctBeat = 0
        print("early stopping!")
        net.load_state_dict(torch.load(args.foldName + '/' + args.modelName + '/' + args.modelName, map_location = args.device))
        
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
                    local_key = keyToOneHot(local_key)
                    output = net(local_batch, local_key)
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
                    local_beat = beatToOneHot(local_beat)
                    output = net(local_batch, local_beat)
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
                    local_key = keyToOneHot(local_key)
                    local_beat = beatToOneHot(local_beat)
                    output = net(local_batch, local_key, local_beat)
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
                    repeat = local_batchRep[i][int(args.lenSeq/decim) - 1]
                    for j in range(int(args.lenPred/decim)):
                        totalTest += 1
                        correctrepeat += (repeat.max(0)[1] == local_labels[i][j].max(0)[1]).item()
                        result = (output[i][j].max(0)[1] == local_labels[i][j].max(0)[1]).item()
                        #correct += result
                        accuraList[j] += result
                        #if args.dist != 'None':
                        #    	musicalD += torch.dot(torch.matmul(output[i][j], distMat), local_labels[i][j])
                lablbatmax = local_labels.data.max(2, keepdim=False)[1]
                pred = output.data.max(2, keepdim=False)[1]
                if args.alphaRep == "alphaRep":
                    batch_correct,_ = chordUtil.computeAccNewRep(args,decim,output,local_labels)
                    correct += batch_correct
                else:
                    correct += pred.eq(lablbatmax.data.view_as(pred)).sum().item()
                
                            
            if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
                # Compute key accuracy
                keybatmax = local_key[:,0].long()
                totalKey += output.size()[0]
                pred = key.data.max(1, keepdim=False)[1]
                correctKey += pred.eq(keybatmax.data.view_as(pred)).sum().item()
                # Compute beat accuracy
                beatbatmax = local_beat[:,0].long()
                totalBeat += output.size()[0]
                pred = beat.data.max(1, keepdim=False)[1]
                correctBeat += pred.eq(beatbatmax.data.view_as(pred)).sum().item()
    
        if args.modelType == "mlpDecimAug" or args.modelType == "mlpDecimAugUp":
            accurKey = 100 * correctKey / totalKey
            accurBeat = 100 * correctBeat / totalBeat
            res["bestAccurKeyTest"] = accurKey
            res["bestAccurBeatTest"] = accurBeat
                            
        if args.decimList[0] == 1:
            accurTest = 100 * correct / totalTest 
            accurRepeat = 100 * correctrepeat / totalTest
            accuraList[:] = [x / (totalTest/(int(args.lenPred/args.decimList[0]))) for x in accuraList]
        print('Best accuracy of the network on test dataset: {} %'.format(accurTest))
        res["bestAccurTest"] = accurTest
        res["repeatAccurTest"] = accurRepeat
        if args.dist != 'None':
            res["musicalDistonTestWithBestValAcc"] = musicalD
        res["bestAccurTestList"] = accuraList
        sauv = open(args.foldName + '/' + args.modelName + '/' + "res" + args.modelName + ".pkl","wb")
        pickle.dump(res,sauv)
        sauv.close()
        print("End of training")
        break  
