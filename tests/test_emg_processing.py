import numpy as np
import pytest

from src.emg_processing import (
    bandpass_filter,
    emg_features,
    moving_rms,
    notch_filter,
    rectify_emg,
)


def test_bandpass_filter_reduces_out_of_band_noise():
    fs = 2000
    t = np.arange(fs * 2) / fs
    signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 600 * t)

    filtered = bandpass_filter(signal, fs)

    assert np.std(filtered) < np.std(signal)
    assert np.sqrt(np.mean(filtered**2)) > 0.5


def test_notch_filter_reduces_line_interference():
    fs = 1000
    t = np.arange(fs * 2) / fs
    signal = np.sin(2 * np.pi * 100 * t) + 0.8 * np.sin(2 * np.pi * 50 * t)

    filtered = notch_filter(signal, fs, line_frequency=50)

    assert np.std(filtered) < np.std(signal)


def test_rectification_returns_absolute_values():
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    np.testing.assert_array_equal(
        rectify_emg(signal),
        np.array([2.0, 1.0, 0.0, 1.0, 2.0]),
    )


def test_moving_rms_is_nonnegative():
    signal = np.sin(2 * np.pi * 100 * np.arange(1000) / 1000)

    result = moving_rms(signal, sampling_rate=1000)

    assert result.shape == signal.shape
    assert np.all(result >= 0)


def test_emg_features_are_correct():
    signal = np.array([-1.0, 1.0, -1.0, 1.0])

    features = emg_features(signal)

    assert features["MAV"] == pytest.approx(1.0)
    assert features["RMS"] == pytest.approx(1.0)
    assert features["Waveform_Length"] == pytest.approx(6.0)
    assert features["Variance"] == pytest.approx(1.0)


def test_bandpass_rejects_invalid_frequency_band():
    signal = np.ones(1000)

    with pytest.raises(ValueError):
        bandpass_filter(signal, sampling_rate=1000, lowcut=20, highcut=600)
