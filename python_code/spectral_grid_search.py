import numpy as np
import math
from scipy.linalg import toeplitz


""" Spectral "grid search" toolbox.
    
    Includes implementations of regular MUSIC and Goertzel's algorithms
    adapted to the context of partials tracking ("spectral grid search").
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



def grid_music(x, p, M, precision_autocorr, fs_recording, freq_res, freq_ranges):
    
    """ Performs a fine "grid search" and extract the local peaks (partials) 
        in specified frequency regions using the regular MUSIC method.
        
        input:
            - x: the signal to estimate the partials from, one-dimensional array
            - p: the assumed number of harmonics
            - M: the size of the autocorrelation matrix to use
            - precision_autocorr: the number of samples to use to
                compute the sample autocorrelation (by default, use the length of x)
            - fs_recording: the sampling frequency of the signal x
            - freq_res: the frequency resolution in Hertz
            - freq_ranges: list of tuples (f_start, f_end) representing
                the frequency intervals to inspect
        
        output:
            - estimated_partials: the estimated partials in the specified frequency
                regions, one-dimensional array
        
    """
    
    # compute the sample autocorrelation
    mid_point = int(np.floor(precision_autocorr/2))
    samp_autocorr = (1/mid_point) * np.correlate(x[0:precision_autocorr], \
        x[0:precision_autocorr], 'same')[mid_point:]
    
    # construct the sample autocorrelation matrix
    Rx = toeplitz(samp_autocorr[0:M], np.conjugate(samp_autocorr[0:M]))
    
    # compute the eigenpairs (eigenvalues, eigenvectors)
    [eig_val, eig_vec] = np.linalg.eigh(Rx)
    
    # sort the eigenvalues in increasing order
    sorted_indices = np.argsort(eig_val)
    
    # form the noise subspace, composed of the last (M-p) eigenvectors
    E = eig_vec[:, sorted_indices[:(M - p)]]
    
    # initializations
    estimated_partials = []
    partials_amps = []
    
    # We iterate over all frequency regions. For each of these, we compute a piece of the spectrum 
    # using MUSIC and extract the local peak (detected partial).
    for freq_range in freq_ranges:
        
        # initializations
        f_start, f_end = freq_range
        freq_grid = np.linspace(f_start, f_end, int((f_end - f_start)/freq_res), False)
        xx, yy = np.meshgrid(freq_grid, np.arange(M))
        
        # frequency evaluation function in the current frequency region
        A = np.exp(-1j* (2*np.pi) * (xx/fs_recording) * yy)
        
        # evaluate a certain frequency region of the spectrum (current range)
        spectrum_slice = - 20 * np.log10(np.linalg.norm(A.T @ E, axis=1))
        
        # extract the local peak and take it as the partial
        estimated_partials.append(freq_grid[np.argmax(spectrum_slice)])
        
        # extract the partial amplitude
        partials_amps.append(spectrum_slice[np.argmax(spectrum_slice)])
        
    return estimated_partials, partials_amps



def goertzel(x, fs_recording, frame_duration, freq_ranges):
    
    """ Performs a "grid search" and extract the local peaks (partials) in specified frequency regions 
        by evaluating the Discrete Fourier Transform (DFT) at selected points with Goertzel's algorithm.
        
        (inspired by https://gist.github.com/sebpiq/4128537)
        
        input:
            - x: the signal to estimate the partials from, one-dimensional array
            - fs_recording: the sampling rate of the recording x
            - frame_duration: the duration of each frame
            - freq_ranges: list of tuples (f_start, f_end) representing
                the frequency intervals to inspect
        
        output:
            - estimated_partials: the estimated partials in the specified frequency
                regions, one-dimensional array
        
    """
    
    # initializations
    window_size = len(x)
    f_step = fs_recording / float(window_size)
    f_step_normalized = 1.0 / window_size
    estimated_partials = []
    partials_amps = []
    
    # calculate all the frequency bins
    bins = list()
    for freq_range in freq_ranges:
        f_start, f_end = freq_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))
        
        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins.append((k_start, k_end))
    
    # for all the bins intervals, calculate the DFT terms
    n_range = range(0, window_size)
    for k_start, k_end in bins:
        
        # initialization: bin frequencies and coefficients
        freqs = np.arange(k_start, k_end) * f_step_normalized
        w_real = 2.0 * np.cos(2.0 * np.pi * freqs)
        w_imag = np.sin(2.0 * np.pi * freqs)
        
        # Goertzel's method
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = x[n] + w_real * d1 - d2
            d2, d1 = d1, y
        
        # store the current powers
        current_powers = d2**2 + d1**2 - w_real * d1 * d2
        
        # => Extract the partial
        # W/o polynomial fit: the partial is chosen as the local maximum in this frequency region
        #estimated_partials.append(fs_recording * freqs[np.argmax(current_powers)])
        
        # W/ polynomial fit: extrapolate to find the spectral peak with more precision
        sorted_freq_idx = np.argsort(freqs)
        sorted_freqs = freqs[sorted_freq_idx] * fs_recording
        sorted_powers = current_powers[sorted_freq_idx]
        arg_local_peak = np.argmax(sorted_powers)
        
        # if on the edge of the interval, use directly the frequency bin
        if(arg_local_peak + 1 > max(sorted_freq_idx)):
            estimated_partials.append(sorted_freqs[arg_local_peak])
        else:
            # otherwise, use the direct upper and lower neighbours to fit a degree-2 polynomial
            # on the peak
            tmp = np.array([arg_local_peak - 1, arg_local_peak, arg_local_peak + 1])
            poly_coef = np.polyfit(sorted_freqs[tmp] * window_size / (frame_duration * fs_recording), \
                         sorted_powers[tmp], 2)
            
            # the estimated partial is then taken as the maximum of the polynomial (zero derivative)
            estimated_partials.append(-poly_coef[1]/(2*poly_coef[0]))
        
        partials_amps.append(sorted_powers[arg_local_peak])
        
    return estimated_partials, partials_amps

