import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_mri(file_path: str) -> tuple[np.ndarray, tuple]:
    """Load a NIfTI MRI volume and return its data and voxel spacing."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"MRI file not found: {path}")

    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)

    volume = np.asarray(image.dataobj, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, but received shape {volume.shape}")

    if not np.isfinite(volume).all():
        raise ValueError("MRI volume contains non-finite values.")

    voxel_spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

    return volume, voxel_spacing


def visualize_mri(file_path: str) -> None:
    """Display the central axial, coronal, and sagittal slices."""
    volume, spacing = load_mri(file_path)

    x, y, z = (dimension // 2 for dimension in volume.shape)

    print(f"Volume dimensions: {volume.shape}")
    print(f"Voxel spacing (mm): {spacing}")
    print(f"Intensity range: {volume.min():.2f} to {volume.max():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    views = [
        (volume[:, :, z].T, "Axial view"),
        (volume[:, y, :].T, "Coronal view"),
        (volume[x, :, :].T, "Sagittal view"),
    ]

    for axis, (slice_data, title) in zip(axes, views):
        axis.imshow(slice_data, cmap="gray", origin="lower")
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle("MRI Volume Viewer")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a 3D NIfTI MRI volume.")
    parser.add_argument("file", help="Path to a .nii or .nii.gz MRI file")
    args = parser.parse_args()

    visualize_mri(args.file)
