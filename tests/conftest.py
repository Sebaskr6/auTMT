"""Shared fixtures and synthetic phantoms for the AuTMT test suite.

The pipeline is normally run on real head CT scans (CQ500). Those cannot be
shipped with the repository, so the tests build small synthetic volumes whose
geometry is known exactly (eyes, skull, temporal muscles) and check that every
stage recovers that geometry.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # the pipeline calls plt.show(); keep tests headless

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# --------------------------------------------------------------------------
# DICOM
# --------------------------------------------------------------------------
def write_dicom_series(folder, n_slices=6, shape=(32, 24), spacing=(0.5, 0.7, 2.0)):
    """Write a small, valid CT DICOM series (one file per axial slice).

    Each slice is air (-1000 HU) with an elliptical 'skull' ring (1500 HU), so
    the bone threshold used by the reorientation step finds something.
    """
    os.makedirs(folder, exist_ok=True)
    uid = "1.2.826.0.1.3680043.2.1125.1.1"
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    ell = ((yy - shape[0] / 2) / (shape[0] * 0.4)) ** 2 + ((xx - shape[1] / 2) / (shape[1] * 0.3)) ** 2
    ring = (ell <= 1.0) & (ell >= 0.6)
    for k in range(n_slices):
        arr = np.full((1,) + shape, -1000, np.int16)
        arr[0][ring] = 1500
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        img.SetOrigin((0.0, 0.0, k * spacing[2]))
        for tag, value in [
            ("0020|000e", uid),                 # Series Instance UID
            ("0020|000d", uid + ".9"),          # Study Instance UID
            ("0008|0018", uid + f".{100 + k}"), # SOP Instance UID
            ("0008|0060", "CT"),
            ("0020|0013", str(k + 1)),          # Instance Number
            ("0018|0050", str(spacing[2])),     # Slice Thickness
        ]:
            img.SetMetaData(tag, value)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(os.path.join(folder, f"slice_{k:03d}.dcm"))
        writer.Execute(img)
    return folder


@pytest.fixture
def dicom_patient(tmp_path):
    """patient/study/series/*.dcm -> returns the patient root folder."""
    write_dicom_series(str(tmp_path / "patient" / "study" / "series"))
    return str(tmp_path / "patient")


# --------------------------------------------------------------------------
# Eye phantom (stage 2)
# --------------------------------------------------------------------------
def eye_phantom(drift=0.0, X=200, Y=40, Z=200, radius=20, eyes_x=(60, 140), eyes_z=140):
    """Binary volume (X, Y, Z) with two eyeballs seen in coronal slices vol[:, i, :].

    ``drift`` moves each eye centre along X by ``drift`` pixels per coronal
    slice, symmetric around the middle slice, so the mean position of each eye
    is still ``eyes_x``. Real eyes are rarely perfectly aligned with the slice
    axis, so this is the realistic case for the tracking step.
    """
    vol = np.zeros((X, Y, Z), np.uint8)
    xx, zz = np.meshgrid(np.arange(X), np.arange(Z), indexing="ij")
    for i in range(Y):
        r = radius * np.sqrt(max(0.0, 1 - ((i - Y / 2) / (Y / 2)) ** 2 * 0.6))
        for cx in eyes_x:
            cxd = cx + drift * (i - Y / 2)
            vol[:, i, :][((xx - cxd) ** 2 + (zz - eyes_z) ** 2) <= r ** 2] = 1
    return vol


# --------------------------------------------------------------------------
# Head phantom (stage 3)
# --------------------------------------------------------------------------
def head_phantom_display(t_left=20, t_right=30, size=300, r_out=110, r_in=100, band_half=40):
    """Axial slice in the *display* frame used by Etapa3.Processing.

    Air (-1000 HU), skull ring (1000 HU), brain (35 HU) and two rectangular
    temporal muscles (60 HU) of ``t_left`` / ``t_right`` pixels, glued to the
    outside of the skull at mid height.
    """
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - c, yy - c)
    img = np.full((size, size), -1000.0)
    img[r <= r_in] = 35.0
    img[(r > r_in) & (r <= r_out)] = 1000.0
    band = np.abs(yy - c) < band_half
    img[band & (xx >= c - r_out - t_left) & (xx < c - r_out)] = 60.0
    img[band & (xx > c + r_out) & (xx <= c + r_out + t_right)] = 60.0
    return img


def save_display_slice_as_nifti(display, path, spacing=(0.5, 0.5)):
    """Store ``display`` so that Processing() reconstructs exactly the same array.

    Processing() applies rot90(k=1) followed by fliplr(); this is the inverse.
    """
    stored = np.rot90(np.fliplr(display), k=-1)
    affine = np.diag([spacing[0], spacing[1], 1.0, 1.0])
    nib.save(nib.Nifti1Image(stored[:, :, None].astype(np.float32), affine), path)
    return path
