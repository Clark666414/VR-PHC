"""
Signal filtering utilities for motion capture data.
Implements Butterworth low-pass filtering and Gaussian smoothing to remove noise and jitter.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


def apply_lowpass_filter(data, cutoff=5.0, fs=30.0, order=4):
    """
    Apply zero-phase Butterworth low-pass filter to remove high-frequency jitter.

    Args:
        data: (N, D) numpy array - time series data
        cutoff: Cutoff frequency in Hz (default: 5.0)
        fs: Sampling rate in Hz (default: 30.0 for Pico VR)
        order: Filter order (default: 4)

    Returns:
        Filtered data with same shape as input
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq

    # Design Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply zero-phase filtering (filtfilt = forward + backward pass)
    # This ensures no time delay in the output
    filtered_data = filtfilt(b, a, data, axis=0)

    return filtered_data


def apply_gaussian_smoothing(data, sigma=3.0):
    """
    Apply Gaussian smoothing along time axis for aggressive anti-jitter.

    Args:
        data: (N, D) numpy array - time series data
        sigma: Gaussian kernel standard deviation (default: 3.0)

    Returns:
        Smoothed data with same shape as input
    """
    if len(data.shape) == 1:
        # 1D data
        return gaussian_filter1d(data, sigma=sigma, axis=0)
    else:
        # Multi-dimensional data - smooth each dimension independently
        smoothed = np.zeros_like(data)
        for dim in range(data.shape[1]):
            smoothed[:, dim] = gaussian_filter1d(data[:, dim], sigma=sigma, axis=0)
        return smoothed


def apply_double_filter(data, butterworth_cutoff=10.0, gaussian_sigma=3.0, fs=30.0):
    """
    Apply double filtering: Butterworth + Gaussian for extreme smoothness.

    This is the "nuclear option" for removing all jitter from optimization outputs.

    Args:
        data: (N, D) numpy array - time series data
        butterworth_cutoff: Butterworth cutoff frequency in Hz (default: 10.0)
        gaussian_sigma: Gaussian smoothing sigma (default: 3.0)
        fs: Sampling rate in Hz (default: 30.0)

    Returns:
        Double-filtered data with same shape as input
    """
    # First pass: Butterworth low-pass filter
    filtered = apply_lowpass_filter(data, cutoff=butterworth_cutoff, fs=fs, order=4)

    # Second pass: Gaussian smoothing
    smoothed = apply_gaussian_smoothing(filtered, sigma=gaussian_sigma)

    return smoothed
