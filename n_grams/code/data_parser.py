import pretty_midi
import numpy as np


quality_map = {'maj'  : 'maj',
               'min'  : 'min',
               'maj7' : 'maj',
               'min7' : 'min',
               'dom7' : 'maj',
               'dim'  : 'dim',
               'd7b6' : 'maj',
               'd7b6 ': 'maj',
               'd7b9' : 'maj',
               'hdim' : 'dim',
               'fdim' : 'dim',
               'fr6'  : 'maj',
               'Fr6'  : 'maj',
               'it6'  : 'maj',
               'It6'  : 'maj',
               'ger6' : 'maj',
               'Ger6' : 'maj'
              }

quality_to_index_map = {'maj' : 0,
                        'min' : 1,
                        'dim' : 2
                       }

tonic_map = {'A' : 9,
             'B' : 11,
             'C' : 0,
             'D' : 2,
             'E' : 4,
             'F' : 5,
             'G' : 7,
             'X' : None,
             'N' : None
            }

accidental_map = {'#' : 1,
                  'b' : -1
                 }


def uniquify(sequence):
    """
    Uniquify a sequence by removing any repeated elements.
    
    Parameters
    ==========
    sequence : iterable
        A sequence with or without consecutive duplicates.
        
    Returns
    =======
    unique : np.array
        A list, equal to the input sequence, but with repeated elements collapsed into a single
        element of that value.
        
    lengths : np.array
        A list of the lengths of each of the returned chords.
    """
    if len(sequence) == 0:
        return np.array([]), np.array([])
    
    lengths = [1]
    unique = [sequence[0]]
    
    for element in sequence[1:]:
        if unique[-1] != element:
            unique.append(element)
            lengths.append(1)
        else:
            lengths[-1] += 1
            
    return np.array(unique), np.array(lengths)


def get_one_hot_index_from_strings(tonic, quality, num_qualities=3):
    """
    Get the one-hot index of a tonic and quality string pair.
    
    Parameters
    ==========
    tonic : string
        A tonic pitch, including any sharps (#) and flats (b).
        
    quality : string
        A chord quality.
        
    num_qualities : int
        The number of allowable qualities in this set of chords. Defaults to 3.
    
    Returns
    =======
    index : int
        The one-hot index of the given chord.
    """
    return get_one_hot_index_from_ints(get_tonic_index(tonic), get_quality_index(quality),
                                       num_qualities=num_qualities)


def get_quality_index(quality):
    """
    Get the index of the quality of a given quality string.
    
    Parameters
    ==========
    quality : string
        A chord quality.
        
    Returns
    =======
    quality_index : int
        The index of the given chord quality.
    """
    try:
        return quality_to_index_map[quality]
    except:
        return None


def get_tonic_index(tonic):
    """
    Get the index of the tonic of the given chord tonic string.
    
    Parameters
    ==========
    tonic : string
        A tonic pitch, including any sharps (#) and flats (b).
        
    Returns
    =======
    tonic_index : int
        The index of the given tonic string, on the range [0 (C), 11 (B)], or None if
        the given tonic string is not a proper pitch.
    """
    try:
        tonic_index = tonic_map[tonic[0]]
        if tonic_index is not None:
            for accidental in tonic[1:]:
                tonic_index += accidental_map[accidental]
            tonic_index %= 12
        
        return tonic_index
    except:
        return None


def transpose(one_hots, diff, num_qualities=3):
    """
    Transpose the input list of one-hot chord indices by diff semitones.
    
    Parameters
    ==========
    one_hots : np.array
        An array of 1-hot chord representations. Integers on the range [0, 12*num_qualities+1].
        
    diff : int
        The amount to transpose each chord by, in semitones.
        
    num_qualities : int
        The number of chord qualities there are in the one-hots. Defaults to 3.
        
    Returns
    =======
    transposed : np.array
        An array of 1-hot chord representations, transposed.
    """
    new_tonics = (one_hots + diff) % 12
    qualities = np.floor(one_hots / 12)
    
    return np.where(one_hots == 12 * num_qualities, 12 * num_qualities, new_tonics + 12 * qualities).astype(int)
    


def get_one_hot_index_from_ints(tonic, quality, num_qualities=3):
    """
    Get the one-hot index of a chord with the given tonic and quality.
    
    Parameters
    ==========
    tonic : int
        The tonic of the chord, where 0 is C. None represents a chord with no tonic.
        
    quality : int
        The quality of the chord, where 0 is major, 1 is minor, and 2 is diminished.
        
    num_qualities : int
        The number of allowable qualities in this set of chords. Defaults to 3.
    
    Returns
    =======
    index : int
        The one-hot index of the given chord and quality. A chord with no tonic returns
        an index of 36.
    """
    try:
        return tonic + 12 * quality
    except:
        return 12 * num_qualities


def get_reduced_chord_sequence(chords, times):
    """
    Get the reduced chord sequence from a list of chords.
    
    Parameters
    ==========
    chords : list
        An ordered list of chords, represented as (tonic, quality) tuples.
        
    times : list
        A list of the times of each chord.
        
    Returns
    =======
    reduced_chords : list
        An ordered list of chords, with consecutive duplicates removed.
        
    reduced_times : list
        A list of the times of each chord in reduced_chords.
    """
    if len(chords) == 0:
        return [], []
    
    reduced_chords = [chords[0]]
    reduced_times = [times[0]]
    
    for time, chord in zip(times[1:], chords[1:]):
        if reduced_chords[-1] != chord:
            reduced_chords.append(chord)
            reduced_times.append(time)
            
    return reduced_chords, reduced_times



def get_sequence_from_kp(midi):
    """
    Get the reduced chord sequence from a kp KP-corpus file.
    
    Parameters
    ==========
    midi : pretty_midi
        A pretty_midi object representing the piece to parse.
        
    Returns
    =======
    chords : list
        The reduced chord sequence from the given piece.
        
    times : list
        The time of each chord in chords.
    """
    def convert_chord_kp(chord):
        """
        Convert the given chord from a string (read from the KP-corpus), to a tonic and quality.

        Parameters
        ==========
        chord : string
            A string representation of a chord.

        Returns
        =======
        tonic : int
            The tonic of the chord, where 0 represents C. A chord with no tonic returns None here.

        quality : int
            The quality of chord, where 0 is major, 1 is minor, and 2 is diminished. Others are None.
        """
        global tonic_map, accidental_map

        chord = chord.split('_')

        tonic = tonic_map[chord[0][0]]
        if tonic is not None:
            for accidental in chord[0][1:]:
                tonic += accidental_map[accidental]
            tonic %= 12
        
        quality = quality_map[chord[1]]

        return tonic, quality
    
    return get_reduced_chord_sequence([convert_chord_kp(lyric.text) for lyric in midi.lyrics],
                                      [lyric.time for lyric in midi.lyrics])
    



def get_chords_and_notes_from_piece(midi):
    """
    Get a list of the chords from a given midi piece.
    
    Parameters
    ==========
    midi : pretty_midi
        The pretty_midi object of a MIDI file.
        
    Returns
    =======
    chords : list
        An ordered list of the (tonic, quality) tuples
    """
    chords, times = get_sequence_from_kp(midi)