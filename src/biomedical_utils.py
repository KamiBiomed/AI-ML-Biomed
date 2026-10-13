import numpy as np


def normalize_signal(signal):
    """Scale signal values to the range [0, 1]."""
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        return []

    min_value = np.min(signal)
    max_value = np.max(signal)

    if min_value == max_value:
        return np.zeros_like(signal).tolist()

    normalized = (signal - min_value) / (max_value - min_value)
    return normalized.tolist()
