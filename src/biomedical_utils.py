import numpy as np


def normalize_signal(signal):
    """Scale a one-dimensional signal to the range [0, 1]."""
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal.size == 0:
        return []

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    min_value = np.min(signal)
    max_value = np.max(signal)

    if min_value == max_value:
        return np.zeros_like(signal).tolist()

    normalized = (signal - min_value) / (max_value - min_value)
    return normalized.tolist()
