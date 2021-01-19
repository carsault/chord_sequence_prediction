import argparse
import kenlm
import numpy as np
import time
import sys
import os
import copy
import itertools
import pickle
import torch

import chordUtil



def get_probs_next_n(model, seed, n, vocab_size, beam=None):
    """
    Get the probability of a each chord for n steps after some seed of chords.
    
    Parameters
    ==========
    model : kenlm.Model
        The model to use for calculating the probabilities.
        
    seed : list
        A list of the chords to initialize the state of the model.
        
    n : int
        The number of steps to generate a probability distribution for.
        
    vocab_size : int
        The number of possible chords in the vocabulary.
        
    beam : int
        The maximum number of states to keep track of at each timestep. Defaults to
        None, which saves all states.
        
    Returns
    =======
    probs : np.ndarray
        A vocab_size X n array, containing the probability of each chord being present
        for n steps after the given seed. The columns will be normalized to sum to 1.
    """
    state = kenlm.State()
    state2 = kenlm.State()
    
    # Initialize model state with seed
    model.BeginSentenceWrite(state)
    for c in seed:
        model.BaseScore(state, str(c), state2)
        state = copy.copy(state2)
    
    probs = np.zeros((vocab_size, n))
    
    # Initialize beam
    states = [state]
    log_probs = [0.0]
    
    for i in range(n):
        new_states = []
        new_log_probs = []
        
        for state, orig_log_prob in zip(states, log_probs):
            for c in range(vocab_size):
                log_prob = orig_log_prob + model.BaseScore(state, str(c), state2)
                
                probs[c, i] += 10 ** log_prob
                
                new_states.append(copy.copy(state2))
                new_log_probs.append(log_prob)
                    
        if beam is not None and len(new_states) > beam:
            states = []
            log_probs = []
            for lp, s in itertools.islice(sorted(zip(new_log_probs, new_states),
                                                 reverse=True), beam):
                states.append(s)
                log_probs.append(lp)
        else:
            states = new_states
            log_probs = new_log_probs
            
    return probs / np.sum(probs, axis=0, keepdims=True)



def get_log_probability(guessed, correct):
    """
    Get the model's probability at each step given the correct sequence.
    
    Parameters
    ==========
    guessed : np.ndarray
        A vocab_size X n array, containing the probability of each chord being present
        at each time step. The columns should be normalized to sum to 1.
        
    correct : list(int)
        The correct one-hot integers.
        
    Returns
    =======
    log_prob : list(float)
        The log probability of the correct output at each step given the model.
    """
    return np.log(np.diag(guessed[correct, :]))



def get_accuracy(guessed, correct):
    """
    Get the accuracy of the guessed sequence given the correct sequence.
    
    Parameters
    ==========
    guessed : list(int)
        The guessed sequence of one-hot integers.
        
    correct : list(int)
        The correct one-hot integers.
        
    Returns
    =======
    accuracy : list(int)
        A binary sequence containing 1 if the guess was correct and 0 otherwise.
    """
    return np.where(guessed == correct, 1, 0)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate some kenlm model on a dataset ' +
                                     'for n -> n chord prediction.')
    
    parser.add_argument('model', help='The KenLM model to use.')
    
    parser.add_argument('test_data', help='The pickle data file to use for testing.')
    
    parser.add_argument("-n", help='The number of chord symbols to generate (and use as a ' +
                        'seed). Defaults to 8.', type=int, default=8)
    
    parser.add_argument('--vocab', help='The chord vocab used to train the given model. ' +
                        'Either a0 (default), a1, a2, a3, or a5. This is used to know how ' +
                        'many possible chord symbols to check for at each step.',
                        choices=['a0', 'a1', 'a2', 'a3', 'a5'], default='a0')
    
    parser.add_argument('--beam', '-b', help='The beam size to use when decoding. Defaults ' +
                        'to None (no beam).', type=int, default=None)
    
    parser.add_argument('--output', help='Directory to save results in. Defaults to None, ' +
                        'which saves nothing. The results pickle file will be named ' +
                        '{output}/{model}.{vocab}.{n}.{beam}.pkl.', default=None)
    
    args = parser.parse_args()
    
    vocabs = {'a0' : chordUtil.a0,
              'a1' : chordUtil.a1,
              'a2' : chordUtil.a2,
              'a3' : chordUtil.a3,
              'a5' : chordUtil.a5}
    vocab_dict = vocabs[args.vocab]
    vocab_size = len(chordUtil.getDictChord(vocab_dict)[0])
    model = kenlm.Model(args.model)
    
    results = np.zeros((0, vocab_size, args.n))
    accuracies = np.zeros((0, args.n))
    log_probs = np.zeros((0, args.n))
    
    with open(args.test_data, "rb") as file:
        pkl = pickle.load(file)
        
    X = pkl['X'].numpy().astype(int)
    Y = pkl['y'].numpy().astype(int)
    key = pkl['key'].numpy().astype(int)
    beat = pkl['beat'].numpy().astype(int)
    start_time = time.time()
    listX = []
    listy = []
    dictDat = {}
    
    listKey = []
    listBeat = []
    print(f"Results for " + args.test_data)
    for i, [x, y, k ,b] in enumerate(zip(X, Y, key, beat)):
        if i % 1000 == 0:
            '''
            print(f"Data point {i+1}/{len(X)} ({100*i/len(X)}%)")
            print(f"Running accuracy: {np.mean(accuracies)}")
            print(f"Running perplexity: {np.exp(-np.mean(log_probs))}")
            print(f"  Time taken so far: {time.time()-start_time}s")
            '''
            sys.stdout.flush()
        if x[0] == vocab_size - 1:
            x = x[len(x) - list(x[::-1]).index(vocab_size - 1):]
        probs = get_probs_next_n(model, x, args.n, vocab_size, beam=args.beam)
        for j in range(len(probs[0])):
            listX.append(probs[:][j])
            listy.append(torch.nn.functional.one_hot(torch.tensor(y[j]).to(torch.int64), len(probs)).float())
            #listKey.append(pkl["key"][i][j])
            #listBeat.append(pkl["beat"][i][j])
            listKey.append(k[j])
            listBeat.append(b[j])
        results = np.vstack((results, np.expand_dims(probs, axis=0)))
        accuracies = np.vstack((accuracies, get_accuracy(np.argmax(probs, axis=0), y)))
        log_probs = np.vstack((log_probs, get_log_probability(probs, y)))

    print(f"Final accuracy: {np.mean(accuracies)}")
    print(f"Final perplexity: {np.exp(-np.mean(log_probs))}")
    print(f"\n\n")
    dictDat["X"] = listX
    dictDat["y"] = listy
    dictDat["key"] = listKey
    dictDat["beat"] = listBeat

    sauv = open(os.path.join(args.output,
                  f"{os.path.basename(args.model)}.{args.vocab}.{args.n}.{args.beam}._test.pkl"),
                  'wb')
    pickle.dump(dictDat,sauv)
    sauv.close()

    if args.output is not None:
        with open(os.path.join(args.output,
                  f"{os.path.basename(args.model)}.{args.vocab}.{args.n}.{args.beam}.pkl"),
                  'wb') as file:
            pickle.dump({'model' : args.model,
                         'vocab' : args.vocab,
                         'n' : args.n,
                         'beam' : args.beam,
                         'results' : results,
                         'accuracy' : accuracies,
                         'log_probs' : log_probs},
                        file)
