def normalize_signal(signal):
    """Scale signal values to the range [0, 1]."""
    if not signal:
        return []

    min_value = min(signal)
    max_value = max(signal)

    if min_value == max_value:
        return [0.0] * len(signal)

    return [(value - min_value) / (max_value - min_value) for value in signal]
