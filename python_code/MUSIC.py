import numpy as np
from scipy.linalg import toeplitz



""" MUSIC (MUltiple SIgnal Classification) toolbox.
    
    Author: Raphaël Latty
    Semester Project - Guessing the fingerings
    EPFL - LCAV - Spring 2017 """



def regular_music(x, p, M, precision_autocorr, NFFT):
    
    """ Implements the regular MUSIC (MUltiple SIgnal Classification) algorithm by constructing 
        a frequency estimation function (estimate of the power spectrum).
        
        (inspired by Statistical Digital Signal Processing and Modeling, Monson H. Hayes, Chapter 8) 
        
        input:
            - x: the signal to estimate the spectrum from, one-dimensional array
            - p: the presumed maximum number of sinusoids
            - M: the size of the autocorrelation matrix to use
            - precision_autocorr: the number of samples to use to 
                compute the sample autocorrelation (by default, use the length of x)
            - NFFT: the number of fft bins/points
        
        output:
            - estimated_spectrum: the estimated pseudo-spectrum, one-dimensional array
        
    """
    
    # compute the sample autocorrelation
    mid_point = int(np.floor(precision_autocorr/2))
    samp_autocorr = (1/mid_point) * np.correlate(x[0:precision_autocorr], \
        x[0:precision_autocorr], 'same')[mid_point:]
    
    # construct the sample autocorrelation matrix
    # (Toeplitz structure from the properties of autocorrelation)
    Rx = toeplitz(samp_autocorr[0:M], np.conjugate(samp_autocorr[0:M]))
    
    # compute the eigenpairs (eigenvalues, eigenvectors)
    [eig_val, eig_vec] = np.linalg.eigh(Rx)
    
    # sort the eigenvalues in increasing order
    sorted_indices = np.argsort(eig_val)
    
    # initialize the estimated spectrum
    estimated_spectrum = 0
    
    # We consider the (M - p) smallest eigenpairs to form the freq. estimation function
    for j in range(M - p):
        
        # construct the estimate step by step
        estimated_spectrum += np.abs(np.fft.fft(eig_vec[:, sorted_indices[j]], NFFT))
        
    # invert the spectrum, square and convert to log scale (dB)
    estimated_spectrum = - (20 * np.log10(estimated_spectrum))
    
    return estimated_spectrum



def root_music(x, fs_recording, p, M, precision_autocorr):
    
    """ Implements the root-MUSIC algorithm by finding the roots of a suitable polynomial
        that are closest to the unit circle.
        
        (inspired by https://github.com/vincentchoqueuse/spectral_analysis_project/)
        
        input:
            - x: the signal to estimate the spectral components from, one-dimensional array
            - fs_recording: the sampling frequency of our recording
            - p: the assumed number of harmonics/spectral components
            - M: the size of the autocorrelation matrix to use
            - precision_autocorr: the number of samples to use to 
                compute the sample autocorrelation (by default use the length of x)
        
        output:
            - estimated_frequencies: the estimated frequency components, one-dimensional array
        
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
    
    # We consider the noise eigenvectors, corresponding to the
    # smallest (M - p) eigenvalues
    V = eig_vec[:, sorted_indices[0:M - p]]
    D = V @ V.conj().T
    
    # construct the polynomial
    Q = 0j * np.zeros(2*M - 1)
    
    # extract the sum in each diagonal
    for (idx, val) in enumerate(range(M - 1, -M, -1)):
        diag = np.diag(D, val)
        Q[idx] = np.sum(diag)
    
    # compute the roots of our polynomial
    roots = np.roots(Q)
    
    # keep the roots with radius < 1 and with non-zero imaginary part
    roots = np.extract(np.abs(roots) < 1, roots)
    roots = np.extract(np.imag(roots) != 0, roots)
    
    # find the p roots closest to the unit circle
    distance_from_circle = np.abs(np.abs(roots) - 1)
    index_sort = np.argsort(distance_from_circle)
    component_roots = roots[index_sort[:p]]
    
    # extract the frequencies (in Hz)
    estimated_angles = np.angle(component_roots)
    estimated_frequencies = fs_recording * estimated_angles/(2*np.pi)
    
    return estimated_frequencies

