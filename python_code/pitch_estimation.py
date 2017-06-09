import numpy as np
from scipy.signal import argrelmax
import collections
import math



""" Pitch estimation method.
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



# dictionary of standard pitches
standard_pitches = {
    'E2': 82.4, 'F2': 87.3, 'F2#': 92.5, 'G2': 98.0, 'G2#': 103.8, 'A2': 110.0, 'A2#': 116.5, 'B2': 123.5, 
    'C3': 130.8, 'C3#': 138.6, 'D3': 146.8, 'D3#': 155.6, 'E3': 164.8, 'F3': 174.6, 'F3#': 185.0, 'G3': 196.0, 
    'G3#': 207.6, 'A3': 220.0, 'A3#': 233.1, 'B3': 246.9, 'C4': 261.6, 'C4#': 277.2, 'D4': 293.7, 'D4#': 311.1,
    'E4': 329.6, 'F4': 349.2, 'F4#': 370.0, 'G4': 392.0, 'G4#': 415.3, 'A4': 440.0, 'A4#': 466.2, 'B4': 493.9, 
    'C5': 523.2, 'C5#': 554.4, 'D5': 587.3, 'D5#': 622.3, 'E5': 659.3
}

# sort in increasing order (by frequency) for further use
standard_pitches = collections.OrderedDict(sorted(standard_pitches.items(), key=lambda t: t[1]))


def estimate_fundamental(estimated_spectrum, fs_recording, freq_res, tol, max_num_peaks):
    
    """ Estimate the pitch/fundamental frequency of a note using Piszczalski's method
        (component frequency ratios).
        
        input:
            - estimated_spectrum: the estimated spectrum
            - fs_recording: the sampling frequency of the recording
            - freq_res: the number of frequency bins (NFFT)
            - tol: a tolerance threshold for discriminating integer frequency ratios
            - max_num_peaks: the maximum number of peaks we consider
        
        output:
            - estimated_fundamental: the estimated fundamental frequency in Hz
        
    """
    
    # extract the spectral peaks from the estimated spectrum (relative maxima)
    spectral_peaks = np.asarray([argrelmax(estimated_spectrum[0:int(freq_res/2)], axis=0, order=10)])
    
    # initialization: candidates for the fundamental frequency (among standard pitches)
    candidates_fund = np.zeros([37, 2])
    candidates_fund[:, 0] = np.fromiter(iter(standard_pitches.values()), dtype=float)
    
    for n in range(max_num_peaks):
        
        # extract the current peak
        current_peak = spectral_peaks[0, 0, n] * fs_recording/freq_res
        
        # for each pair (current_peak, next_peak), try to find the smallest harmonic numbers
        # that would correspond to a harmonic series with these peaks in it (approximately)
        for m in np.arange(n + 1, max_num_peaks + 1):
            
            # initializations
            curr_min_ratio_1 = 100
            curr_min_ratio_2 = 100
            best_candidate = 0
            
            # extract the next peak to form the pair
            next_peak = spectral_peaks[0, 0, m] * fs_recording/freq_res
            
            # search through the dictionary of standard pitches the most likely
            # fundamental frequency f_0
            for pitch, freq in standard_pitches.items():
                
                # break if we reached the current peak
                if(freq > current_peak + 20): break
                
                # evaluate the frequency ratios 
                freq_ratio_current = current_peak/freq
                freq_ratio_next = next_peak/freq
                
                # condition for harmonicity: frequency ratios must be almost integer
                # multiples of the current candidate for the fundamental, i.e small
                # decimal part (up to some tolerance)
                cond_1 = np.abs(math.modf(freq_ratio_current)[0]) < tol
                cond_2 = np.abs(math.modf(freq_ratio_next)[0]) < tol
                cond_3 = np.abs(math.modf(freq_ratio_current)[0]) > (1 - tol)
                cond_4 = np.abs(math.modf(freq_ratio_next)[0]) > (1 - tol)
                
                # if one of these combinations is satisfied (both are almost integer mupltiples of the fundamental)
                if((cond_1 and cond_2) or (cond_1 and cond_4) or (cond_2 and cond_3) or (cond_3 and cond_4)):
                    
                    # we take the SMALLEST harmonic numbers
                    if(np.round_(freq_ratio_current) <= curr_min_ratio_1 and np.round_(freq_ratio_next) <= curr_min_ratio_2):
                        curr_min_ratio_1 = np.round_(freq_ratio_current)
                        curr_min_ratio_2 = np.round_(freq_ratio_next)
                        best_candidate = freq
            
            # compute the weights
            weight = (estimated_spectrum[spectral_peaks[0, 0, n]] + estimated_spectrum[spectral_peaks[0, 0, m]])/2
            #weight = 0
            
            # "vote" for the best candidate with appropriate weights
            candidates_fund[np.where(candidates_fund[:, 0] == best_candidate), 1] += 10**(weight/10)            
           
    # estimate the fundamental frequency based on the partials pairs votes
    estimated_fundamental = candidates_fund[np.argmax(candidates_fund[:, 1]), 0]
    
    return estimated_fundamental

