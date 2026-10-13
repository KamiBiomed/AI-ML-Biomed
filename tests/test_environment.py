import numpy as np
import torch


def test_numpy():
    array = np.array([1, 2, 3])
    assert array.sum() == 6


def test_pytorch():
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6


from src.copilot_test import calculate_mean


def test_calculate_mean():
    assert calculate_mean([1, 2, 3]) == 2


def test_calculate_mean_decimal_values():
    assert calculate_mean([1, 2, 4]) == 7 / 3


def test_calculate_mean_empty_list():
    assert calculate_mean([]) == 0
