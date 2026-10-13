import numpy as np
import pytest
import torch

from src.biomedical_utils import normalize_signal
from src.copilot_test import calculate_mean


def test_numpy():
    array = np.array([1, 2, 3])
    assert array.sum() == 6


def test_pytorch():
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6


def test_calculate_mean():
    assert calculate_mean([1, 2, 3]) == 2


def test_calculate_mean_decimal_values():
    assert calculate_mean([1, 2, 4]) == 7 / 3


def test_calculate_mean_empty_list():
    assert calculate_mean([]) == 0


def test_normalize_signal():
    result = normalize_signal([2, 4, 6])
    assert result == [0.0, 0.5, 1.0]


def test_normalize_signal_empty():
    assert normalize_signal([]) == []


def test_normalize_signal_constant():
    assert normalize_signal([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_normalize_signal_numpy_array():
    signal = np.array([-2, 0, 2, 4])

    result = normalize_signal(signal)

    assert np.allclose(
        result,
        [0.0, 1 / 3, 2 / 3, 1.0],
    )


def test_normalize_signal_rejects_nan():
    with pytest.raises(ValueError, match="finite values"):
        normalize_signal([1, float("nan"), 3])


def test_normalize_signal_rejects_infinity():
    with pytest.raises(ValueError, match="finite values"):
        normalize_signal([1, float("inf"), 3])


def test_normalize_signal_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_signal([[1, 2], [3, 4]])
