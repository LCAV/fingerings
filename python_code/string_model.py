import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as pltlab
import matplotlib.image as img


""" String-related utilities.
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



def compute_string_length(string_full_length, fret_pos):
    
    """ Compute the resulting string length given the full length of the string
        and a fretboard position. 
        
        input:
            - string_full_length: the original full length of the string
            - fret_pos: the position on the fretboard
        
        output:
            - curr_string_length: the corresponding string length
         
    """
    
    current_length = string_full_length
    
    # slightly more precise than the famous "Rule of Eighteen"
    # for placing the frets on a guitar fingerboard
    for n in range(fret_pos):
        current_length -= (1/17.817) * current_length
    
    return current_length



def create_string_motion(modes_num, max_mode, init_displacement, plucking_point, ref_point, f0, c, t):
    
    """ Compute the composite string motion by summing up the vibrational modes with 
        appropriate amplitudes.
        
        input:
            - modes_num: the modes serial numbers
            - max_mode: the maximum mode we consider
            - init_displacement: the initial string displacement
            - plucking_point: the plucking point (in fraction of the length)
            - ref_point: the reference point at which we compute the string motion
            - f0: the fundamental frequency
            - c: the wave speed
            - t: the time indices
        
        output:
            - mode_amplitudes: the amplitudes of the various modes
            - string_motion: the string motion
        
    """
    
    # compute the mode amplitudes from the initial displacement and plucking point
    mode_amplitudes = ((1/plucking_point)**2*init_displacement)/(2*modes_num**2*np.pi**2) * \
        np.sin(modes_num*np.pi*plucking_point)
    
    # compute the composite string motion at the reference point
    string_motion = np.zeros(t.shape)
    for n in np.arange(max_mode):
        string_motion += mode_amplitudes[n] * np.sin(2 * np.pi * (f0*(n + 1) * t)) * \
            np.sin((2 * np.pi * f0 *(n + 1) * ref_point)/c) * np.exp(- 2 * np.pi**2 * np.sqrt(f0 * (n + 1)) * 1e-3 * t)
    
    return mode_amplitudes, string_motion



def plot_string_motion(f0, string_motion, modes_amplitudes, t, max_mode, modes_num, NFFT, f_samp, noverlap):
    
    """ Visualization method: plot the string motion, the relative modes' amplitudes
        and the spectrogram.
        
        input:
            - f0: the fundamental frequency of the note
            - string_motion: the calculated string motion
            - modes_amplitudes: the calculated modes' amplitudes
            - t: the time indices
            - max_mode: the maximum mode number we consider
            - modes_num: the modes serial numbers
            - NFFT: the frequency resolution of the spectrogram
            - f_samp: the sampling rate
            - noverlap: the amount of overlap of the spectrogram
        
        output: none
        
    """
    
    plt.figure(figsize=(12, 8))
    
    # plot the string motion
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    ax1.plot(t, string_motion)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('String motion')
    
    # plot the relative modes' amplitudes
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    ax2.stem(modes_num, modes_amplitudes/max(modes_amplitudes))
    ax2.plot(np.arange(0, max_mode + 1), np.zeros(max_mode + 1), 'k')
    plt.xlabel('Partial number')
    plt.ylabel('Relative amplitude')
    plt.title('Modes relative amplitudes')
    plt.xlim(0, max_mode)
    
    # plot the spectrogram
    ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    specgram, f_bins, t_bins, im = plt.specgram(string_motion, NFFT, f_samp, noverlap=noverlap)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of the string motion')
    plt.xlim(0, 5)
    plt.ylim(0, (max_mode + 1) * f0)
    
    plt.tight_layout()
    plt.show()
    
    return



def show_freboard(fingering):
    
    """ Visualziation method: plots an image of a blank fretboard and draw a red dot to indicate
        the current fingering.
        
        input:
            - fingering: the fingering to plot on the fretboard image,
                (string_number, fretboard_position) pair
        
        output: none
        
    """
    
    # extract the fingering details
    str_num, fret_pos = fingering
    
    # read the blank fingerboard image
    im = img.imread('images_animations/guitar_fretboard_blank.png')
    
    plt.figure(figsize=(12, 8))
    
    # remove the axes
    ax1 = plt.axes(frameon=False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    
    # plot the blank fretboard
    implot = plt.imshow(im)
    
    y_dot = 134 - (str_num - 1) * 24
    
    if(fret_pos == 0):
        x_dot = 14
    else:
        x_dot = 36 + (fret_pos - 1) * 48.66
    
    # draw a red dot to indicate the position on the neck
    # NOTE: numbers were chosen manually
    plt.scatter([x_dot], [y_dot], c='r', s=200)
    plt.tight_layout()
    plt.show()
    
    return



def compare_fingerings(fingering_1, fingering_2, string_length, original_pluck, init_displacement, \
                      ref_point, velocities, t, modes_num, max_mode, NFFT, f_samp, noverlap):
    
    """ Comparison of two fingerings: plot the relative modes' amplitudes and spectrograms of
        both fingerings.
        
        input:
            - fingering_1: first fingering (str_num_1, fret_pos_1)
            - fingering 2: second fingering (str_num_2, fret_pos_2)
            - string_length: the full length of the string
            - original_pluck: the original plucking point (in fraction of the full length)
            - init_displacement: the initial displacement
            - ref_point: the reference point
            - velocities: the wave velocities for all the strings
            - t: the time indices
            - modes_num: the modes serial numbers
            - max_mode: the maximum mode number we consider
            - NFFT: the frequency resolution of the spectrogram
            - f_samp: the sampling rate
            - noverlap: the amount of overlap in the spectrogram
            
        output: none
    
    """
    
    # compute the effective string lengths of both fingerings
    string_length_1 = compute_string_length(string_length, fingering_1[1])
    string_length_2 = compute_string_length(string_length, fingering_2[1])
    
    # recompute the new plucking points with respect to the current length
    plucking_point_1 = original_pluck * string_length / string_length_1
    plucking_point_2 = original_pluck * string_length / string_length_2
    
    # extract the velocities
    c1 = velocities[fingering_1[0]]
    c2 = velocities[fingering_2[0]]
    
    # compute the corresponding fundamental frequencies
    f0_1 = c1/(2*string_length_1)
    f0_2 = c2/(2*string_length_2)
    
    # compute the string motions for both fingerings
    modes_amplitudes_1, string_motion_1 = create_string_motion(modes_num, max_mode, init_displacement, \
                                           plucking_point_1, ref_point, f0_1, c1, t)
    modes_amplitudes_2, string_motion_2 = create_string_motion(modes_num, max_mode, init_displacement, \
                                           plucking_point_2, ref_point, f0_2, c2, t)
    
    # plot the results
    plt.figure(figsize=(12, 8))
    
    # fingering 1
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    ax1.stem(modes_num, modes_amplitudes_1/max(modes_amplitudes_1))
    ax1.plot(np.arange(0, max_mode + 1), np.zeros(max_mode + 1), 'k')
    plt.xlabel('Partial number')
    plt.ylabel('Relative amplitude')
    plt.title('Fingering 1: Modes relative amplitudes')
    plt.xlim(0, max_mode)

    ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    specgram, f_bins, t_bins, im = plt.specgram(string_motion_1, NFFT, f_samp, noverlap=noverlap)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Fingering 1: Spectrogram of the string motion')
    plt.xlim(0, 5)
    plt.ylim(0, (max_mode + 1) * f0_1)
    
    # fingering 2
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
    ax3.stem(modes_num, modes_amplitudes_2/max(modes_amplitudes_2))
    ax3.plot(np.arange(0, max_mode + 1), np.zeros(max_mode + 1), 'k')
    plt.xlabel('Partial number')
    plt.ylabel('Relative amplitude')
    plt.title('Fingering 2: Modes relative amplitudes')
    plt.xlim(0, max_mode)
    
    ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
    specgram, f_bins, t_bins, im = plt.specgram(string_motion_2, NFFT, f_samp, noverlap=noverlap)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Fingering 2: Spectrogram of the string motion')
    plt.xlim(0, 5)
    plt.ylim(0, (max_mode + 1) * f0_2)
    plt.tight_layout()
    plt.show()
    
    return 
