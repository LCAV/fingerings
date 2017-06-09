import numpy as np
from spectral_grid_search import *


""" Partials tracking method.
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



def partials_tracking(recording, fs_recording, frame_duration, estimated_fundamental, \
                      max_mode, delta_f, method, M, freq_res):
    
    """ Track the partials through successive time frames of a specified duration, given an 
        estimated pitch and a method to use (Goertzel or MUSIC).
        
        input:
            - recording: the recording to process, one-dimensional array
            - fs_recording: the sampling rate of the recording
            - frame_duration: the duration of each frame in seconds
            - estimated_fundamental: the estimated fundamental frequency
            - max_mode: the maximum partial number we consider
            - delta_f: the frequency interval half-width to search around the harmonics
            - method: the type of method to use, either goertzel or MUSIC
            - M: the size of the correlation matrix for the modified MUSIC
            - freq_res: the frequency resolution for the modified MUSIC
        
        output:
            - estimated_partials: the estimated partials in all the frames,
                with shape (max_mode, num_frames)
            - partials_amplitudes: the estimated partials amplitudes,
                with shape (max_mode, num_frames)
        
    """
    
    # initializations
    tot_num_samp = recording.shape[0]
    num_samp_per_frame = int(fs_recording * frame_duration)
    window = np.hamming(num_samp_per_frame)
    hop_size = int(num_samp_per_frame/2)
    num_frames = int((tot_num_samp - num_samp_per_frame)/hop_size)
    
    estimated_partials = np.zeros([max_mode, num_frames])
    partials_amplitudes = np.zeros([max_mode, num_frames])
    
    freq_ranges = [(int(k*estimated_fundamental - k*delta_f), int(k*estimated_fundamental + k*delta_f)) \
               for k in range(1, max_mode + 1)]
    
    # analysis of successive frames
    for n in range(num_frames):
        
        # extract the current time frame (windowed block)
        curr_frame = recording[n*hop_size:n*hop_size + num_samp_per_frame] * window
        
        if(method == 'MUSIC'):
            # METHOD 1: estimate the partials using our modified MUSIC algorithm
            estimated_partials[:, n], partials_amplitudes[:, n] = grid_music(curr_frame, 5 * max_mode, M, len(curr_frame), \
                fs_recording, freq_res, freq_ranges)
        else:
            # METHOD 2 (default): estimate the partials using Goertzel's algorithm
            estimated_partials[:, n], partials_amplitudes[:, n] = goertzel(curr_frame, fs_recording, frame_duration, freq_ranges)
    
    
    return estimated_partials, partials_amplitudes

