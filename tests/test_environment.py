import numpy as np
import torch


def test_numpy():
    array = np.array([1, 2, 3])
    assert array.sum() == 6


def test_pytorch():
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6
