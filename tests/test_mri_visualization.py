import nibabel as nib
import numpy as np
import pytest

from src.mri_visualization import load_mri


def test_load_mri_returns_volume_and_spacing(tmp_path):
    data = np.zeros((8, 10, 12), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(data, affine)
    file_path = tmp_path / "sample.nii.gz"
    nib.save(image, str(file_path))

    volume, spacing = load_mri(str(file_path))

    assert volume.shape == (8, 10, 12)
    assert spacing == (2.0, 2.0, 2.0)


def test_load_mri_rejects_two_dimensional_image(tmp_path):
    data = np.zeros((8, 10), dtype=np.float32)
    image = nib.Nifti1Image(data, np.eye(4))
    file_path = tmp_path / "two_dimensional.nii.gz"
    nib.save(image, str(file_path))

    with pytest.raises(ValueError, match="Expected a 3D MRI volume"):
        load_mri(str(file_path))


def test_load_mri_rejects_missing_file(tmp_path):
    file_path = tmp_path / "missing.nii.gz"

    with pytest.raises(FileNotFoundError):
        load_mri(str(file_path))
