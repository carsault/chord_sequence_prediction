import argparse
import os
import sys
import glob
import numpy as np

from sklearn.model_selection import train_test_split

import data_parser
import chordUtil
from chordUtil import *


def get_chords(filename, vocab='a0'):
    """
    Get a list of the chords (as strings) in a given file, with 1 symbol per beat.
    
    Parameters
    ==========
    filename : string
        The full filename of an xlab file to parse.
        
    vocab : string
        The name of the vocabulary to use, from chordUtil. Defaults to a0.
        
    Returns
    =======
    chords : list(string)
        A list of chords, 1 symbol per beat, from the given pieces. Each chord string is
        of the form tonic:quality.
    """
    chords = []
    
    with open(filename, 'r') as file:
        for line in file.read().split('\n')[1:]:
            split = line.split()
            chords.extend([chordUtil.reduChord(split[4], alpha=vocab)] * int(split[2]))
            
    return chords



def get_one_hot(chord, qualities):
    """
    Get the one hot index of a chord string, given the list of possible qualities.
    
    Parameters
    ==========
    chord : string
        The string representation of a chord, in the form tonic:quality.
        
    qualities : list(string)
        A list of the allowable qualities for the given chord.
        
    Returns
    =======
    one_hot : integer
        The one hot index of the given chord string, where C:quality[0]=0,
        C:quality[1]=12, etc. If the given chord has no tonic, or is of an
        unrecognized quality, the returned value is the no chord symbol,
        12*len(qualities).
    """
    
    #try:
    #    tonic, quality = chord.split(':')
    #    quality_index = qualities.index(quality)
    #    tonic_index = data_parser.get_tonic_index(tonic)
    #except:
    #    return 12 * len(qualities)
    
    #return 12 * quality_index + tonic_index
    
    return dictChord[chord]





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and print chords as one-hot ' +
                                     'index sequences to std out.')
    
    parser.add_argument('data_dir', help='The directory containing all of the .xlab ' +
                        'data files to parse.')
    
    parser.add_argument("--split", help='The split to print. Either train, test, valid, ' +
                        'or all (default).', choices=["train", "test", "valid", "all"],
                        default='all')
    
    parser.add_argument("--seed", help='The random seed to use when splitting the files. ' +
                        'Defaults to 123456.', type=int, default=123456)
    
    parser.add_argument('--vocab', help='The chord vocab to use. Either a0 (default), ' +
                        'a1, a2, a3, or a5.', choices=['a0', 'a1', 'a2', 'a3', 'a5'],
                        default='a0')
    parser.add_argument('--transpose', help='Transpose each piece into every other key ' +
                        'before printing it.', action='store_true')
    parser.add_argument('--unique', help='Remove repeated chords before printing.',
                        action='store_true')
    
    parser.add_argument('--vocab_only', help='Print only the vocab list rather than ' +
                        'parsing any of the xlab files. This is useful for creating a ' +
                        'vocab.txt file to use when training a kenlm model.',
                        action='store_true')
    
    args = parser.parse_args()
    
    filenames = sorted(glob.glob(os.path.join(args.data_dir, '*.xlab')))
    
    if args.split != 'all':
        # Create splits
        files_train, files_test = train_test_split(filenames, test_size=0.2, random_state=args.seed)
        files_test, files_valid = train_test_split(files_test, test_size=0.5, random_state=args.seed)
        
        if args.split == 'test':
            filenames = files_test
        elif args.split == 'train':
            filenames = files_train
        elif args.split == 'valid':
            filenames = files_valid
            
    # Here, filenames contains the files to parse and print
    
    vocabs = {'a0' : chordUtil.a0,
              'a1' : chordUtil.a1,
              'a2' : chordUtil.a2,
              'a3' : chordUtil.a3,
              'a5' : chordUtil.a5}
    vocab_dict = vocabs[args.vocab]
    
    qualities = sorted(list(set(vocab_dict.values()) - set(['N'])))

    dictChord, listChord = chordUtil.getDictChord(eval(args.vocab))
    
    if args.vocab_only:
        print(' '.join(str(c) for c in range(12 * len(qualities) + 1)))
        sys.exit(0)
    
    for file in filenames:
        chords = get_chords(file, vocab=args.vocab)
        #one_hots = np.array([get_one_hot(c, qualities) for c in chords])
        one_hots = np.array([dictChord[c] for c in chords])
        
        if args.unique:
            one_hots, _ = data_parser.uniquify(one_hots)
            
        if args.transpose:
            for t in range(12):
                print(' '.join(str(c) for c in data_parser.transpose(one_hots, t,
                                                                     num_qualities=len(qualities))))
        else:
            print(' '.join(str(c) for c in one_hots))