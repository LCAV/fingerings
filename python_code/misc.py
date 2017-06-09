import numpy as np
import collections
import numpy.matlib
from string_model import *


""" Miscellaneous utilities.
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



# fundamental-to-fingering mappings
fundamental_to_fingerings = {
    82.4: [(6, 0)], 87.3: [(6, 1)], 92.5: [(6, 2)], 98.0: [(6, 3)], 103.8: [(6, 4)], 110.0: [(6, 5), (5, 0)],
    116.5: [(6, 6), (5, 1)], 123.5: [(6, 7), (5, 2)], 130.8: [(6, 8), (5, 3)], 138.6: [(6, 9), (5, 4)],
    146.8: [(6, 10), (5, 5), (4, 0)], 155.6: [(6, 11), (5, 6), (4, 1)], 164.8: [(6, 12), (5, 7), (4, 2)],
    174.6: [(5, 8), (4, 3)], 185.0: [(5, 9), (4, 4)], 196.0: [(5, 10), (4, 5), (3, 0)], 
    207.6: [(5, 11), (4, 6), (3, 1)], 220.0: [(5, 12), (4, 7), (3, 2)], 233.1: [(4, 8), (3, 3)],
    246.9: [(4, 9), (3, 4), (2, 0)], 261.6: [(4, 10), (3, 5), (2, 1)], 277.2: [(4, 11), (3, 6), (2, 2)],
    293.7: [(4, 12), (3, 7), (2, 3)], 311.1: [(3, 8), (2, 4)], 329.6: [(3, 9), (2, 5), (1, 0)],
    349.2: [(3, 10), (2, 6), (1, 1)], 370.0: [(3, 11), (2, 7), (1, 2)], 392.0: [(3, 12), (2, 8), (1, 3)],
    415.3: [(2, 9), (1, 4)], 440.0: [(2, 10), (1, 5)], 466.2: [(2, 11), (1, 6)], 493.9: [(2, 12), (1, 7)],
    523.2: [(1, 8)], 554.4: [(1, 9)], 587.3: [(1, 10)], 622.3: [(1, 11)], 659.3: [(1, 12)]
}



def estimate_inharmonicity_poly(num_recordings, stacked_freq_ratios, num_frames, max_mode):
    
    """ Compute the inharmonicity coefficient from the estimated partials in successive time
        frames by fitting a degree-4 polynomial to the stacked, squared frequency ratios.
        
        input:
            - num_recordings: the number of recordings used to estimate the inharmonicity
            - stacked_freq_ratios: stacked, squared frequency ratios (left-hand side vector of the system) 
            - num_frames: the number of frames per recording
            - max_mode: the maximum partial number we consider
        
        output:
            - b_hat: the estimated inharmonicity constant
    """
    
    # initialize the system
    LHS = np.hstack(stacked_freq_ratios)
    system_matrix = np.matlib.repmat(np.vstack((np.arange(1, max_mode + 1)**2, \
                                            np.arange(1, max_mode + 1)**4)).T, num_recordings*num_frames, 1)
    
    # 2D least squares fit (polynomial fit)
    coef, _, _, _ = np.linalg.lstsq(system_matrix, LHS)
    b_hat = coef[1]
    
    return b_hat



def find_best_fingering(gammas, estimated_fundamental, b_hat, string_full_length):
    
    """ Finds the best fingering for the estimated fundamental and inharmonicity constant.
        
        input:
            - gammas: the output of our training phase (inharmonicity profiles of each string)
            - estimated_fundamental: the estimated fundamental frequency of the test recording
            - b_hat: the estimated inharmonicity constant of the test recording
            - string_full_length: the full length of the strings
        
        output:
            - fingerings_candidates: the possible fingerings for the estimated fundamental
            - b_candidates: the inharmonicity of the potential fingering candidates
            - best_fingering: a pair (string_number, fretboard_position) corresponding to 
                the best fingering (closest inharmonicity)
    """
    
    # initializations
    fingerings_candidates = []
    gamma_candidates = []
    lengths_candidates = []
    
    # search in our dictionary of fundamental-to-fingering mappings
    if estimated_fundamental in fundamental_to_fingerings.keys():
        fingerings_candidates = fundamental_to_fingerings[estimated_fundamental]
    else:
        print('Error: Fundamental not found!')
        return
    
    print("\n\t\tPotential fingerings: {fing_cand}".format(fing_cand=fingerings_candidates))
    
    # for all potential fingerings
    for fingering in fingerings_candidates:
        string_num, fret_pos = fingering
        gamma_candidates.append(gammas[string_num - 1])
        lengths_candidates.append(compute_string_length(string_full_length, fret_pos))
    
    # compute the inharmonicity constants corresponding to all the possible fingerings
    b_candidates = np.array(gamma_candidates)/np.array(lengths_candidates)**2
    print("\t\tCorresponding inharmonicity constants: {bs}\n".format(bs=b_candidates))
    
    # find the "best" fingering (closest inharmonicity constant)
    best_fingering_idx = np.argmin(np.abs(b_candidates - b_hat))
    best_fingering = fingerings_candidates[best_fingering_idx]
    
    return fingerings_candidates, b_candidates, best_fingering

