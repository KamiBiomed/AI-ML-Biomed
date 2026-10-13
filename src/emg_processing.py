"""Core preprocessing and feature extraction for EMG signals."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def validate_emg(signal, sampling_rate):
    """Validate a one-dimensional EMG signal and its sampling rate."""
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("EMG signal must be one-dimensional.")
    if x.size < 3:
        raise ValueError("EMG signal must contain at least 3 samples.")
    if not np.isfinite(x).all():
        raise ValueError("EMG signal contains non-finite values.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive and finite.")

    return x


def bandpass_filter(signal, sampling_rate, lowcut=20.0, highcut=450.0, order=4):
    """Retain the selected EMG frequency band."""
    x = validate_emg(signal, sampling_rate)
    nyquist = sampling_rate / 2

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError("Require 0 < lowcut < highcut < sampling_rate / 2.")

    sos = butter(
        order, [lowcut, highcut], btype="bandpass", fs=sampling_rate, output="sos"
    )
    return sosfiltfilt(sos, x)


def notch_filter(signal, sampling_rate, line_frequency=50.0, quality=30):
    """Suppress 50 Hz or 60 Hz electrical interference."""
    x = validate_emg(signal, sampling_rate)

    if not 0 < line_frequency < sampling_rate / 2:
        raise ValueError("Line frequency must be below Nyquist frequency.")
    if quality <= 0:
        raise ValueError("Quality factor must be positive.")

    b, a = iirnotch(line_frequency, quality, fs=sampling_rate)
    return filtfilt(b, a, x)


def rectify_emg(signal):
    """Full-wave rectify an EMG signal."""
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Provide a non-empty, finite 1D signal.")

    return np.abs(x)


def moving_rms(signal, sampling_rate, window_seconds=0.05):
    """Calculate moving RMS using a window, defaulting to 50 ms."""
    x = validate_emg(signal, sampling_rate)

    if window_seconds <= 0 or not np.isfinite(window_seconds):
        raise ValueError("Window duration must be positive and finite.")

    window_samples = max(1, round(window_seconds * sampling_rate))
    mean_square = uniform_filter1d(x**2, size=window_samples, mode="nearest")
    return np.sqrt(np.maximum(mean_square, 0.0))


def emg_features(signal):
    """Extract basic time-domain features from one EMG segment."""
    x = validate_emg(signal, sampling_rate=1.0)

    return {
        "MAV": float(np.mean(np.abs(x))),
        "RMS": float(np.sqrt(np.mean(x**2))),
        "Waveform_Length": float(np.sum(np.abs(np.diff(x)))),
        "Variance": float(np.var(x)),
    }
