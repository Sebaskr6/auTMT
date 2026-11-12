"""Stage 1: DICOM loading and PCA-based head reorientation (Etapa1.py)."""

import os

import numpy as np
import pytest
import SimpleITK as sitk

import Etapa1
from Etapa1 import (
    aplicar_rotacion_en_header,
    cargar_volumen_estandar,
    find_dicom_series_dir,
    guardar_volumen_como_nii,
    harden_to_identity,
    obtener_matriz_orientacion,
    obtener_rotacion_eje_x,
    obtener_rotacion_eje_y,
    obtener_rotacion_eje_z,
    proyectar_plano,
)

ROTATIONS = [obtener_rotacion_eje_x, obtener_rotacion_eje_y, obtener_rotacion_eje_z]
ANGLES = [-170.0, -45.0, -7.5, 0.0, 3.0, 30.0, 90.0, 180.0]


# ---------------------------------------------------------------- rotations
@pytest.mark.parametrize("rot", ROTATIONS)
@pytest.mark.parametrize("angle", ANGLES)
def test_rotation_matrices_are_proper_rotations(rot, angle):
    R = rot(angle)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("rot", ROTATIONS)
def test_zero_angle_is_identity(rot):
    np.testing.assert_allclose(rot(0.0), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("rot", ROTATIONS)
def test_opposite_angles_cancel_and_angles_compose(rot):
    np.testing.assert_allclose(rot(25.0) @ rot(-25.0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rot(20.0) @ rot(15.0), rot(35.0), atol=1e-12)


@pytest.mark.parametrize("rot, axis", list(zip(ROTATIONS, range(3))))
def test_rotation_leaves_its_own_axis_fixed(rot, axis):
    e = np.zeros(3)
    e[axis] = 1.0
    np.testing.assert_allclose(rot(37.0) @ e, e, atol=1e-12)


def test_positive_angle_rotates_clockwise_seen_from_positive_axis():
    # The implementation uses t = -radians(angle): +90 deg about Z sends +X to -Y.
    # This is the convention the header correction relies on (it undoes the
    # measured skull tilt), so it is locked here.
    np.testing.assert_allclose(obtener_rotacion_eje_z(90.0) @ [1, 0, 0], [0, -1, 0], atol=1e-12)


# --------------------------------------------------------- header rotation
def _volume(size=(20, 16, 12), spacing=(0.8, 0.8, 2.0), origin=(-10.0, 5.0, 30.0)):
    arr = np.random.default_rng(0).integers(-1000, 1500, size[::-1]).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def _physical_center(img):
    idx = ((np.array(img.GetSize(), float) - 1) / 2.0).tolist()
    return np.array(img.TransformContinuousIndexToPhysicalPoint(idx))


def test_header_rotation_keeps_voxels_and_rotates_about_center():
    img = _volume()
    R = obtener_rotacion_eje_z(12.0) @ obtener_rotacion_eje_x(-6.0)
    out = aplicar_rotacion_en_header(img, R)

    np.testing.assert_array_equal(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(img))
    assert out.GetSpacing() == img.GetSpacing()
    np.testing.assert_allclose(_physical_center(out), _physical_center(img), atol=1e-6)
    np.testing.assert_allclose(obtener_matriz_orientacion(out), R @ obtener_matriz_orientacion(img), atol=1e-9)


def test_header_rotation_with_identity_changes_nothing():
    img = _volume()
    out = aplicar_rotacion_en_header(img, np.eye(3))
    np.testing.assert_allclose(out.GetOrigin(), img.GetOrigin())
    np.testing.assert_allclose(out.GetDirection(), img.GetDirection())


def test_header_rotation_does_not_modify_input():
    img = _volume()
    before = (img.GetOrigin(), img.GetDirection())
    aplicar_rotacion_en_header(img, obtener_rotacion_eje_y(20.0))
    assert (img.GetOrigin(), img.GetDirection()) == before


def test_harden_to_identity_on_rotated_volume():
    img = aplicar_rotacion_en_header(_volume(), obtener_rotacion_eje_z(30.0))
    out = harden_to_identity(img)
    np.testing.assert_allclose(out.GetDirection(), np.eye(3).ravel(), atol=1e-12)
    assert out.GetSpacing() == img.GetSpacing()
    # A rotated box needs a larger axis-aligned bounding box, never a smaller one.
    assert all(o >= i for o, i in zip(out.GetSize(), img.GetSize()))


def test_harden_to_identity_is_noop_for_aligned_volume():
    img = _volume()
    out = harden_to_identity(img)
    assert out.GetSize() == img.GetSize()
    np.testing.assert_allclose(out.GetOrigin(), img.GetOrigin())
    # Linear resampling of int16 data may truncate by 1 HU; anything larger is a real shift.
    np.testing.assert_allclose(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(img), atol=1)


# ------------------------------------------------------------- PCA angle
def _bar_volume(angle_deg, plane):
    """Thin 'bone' bar (2000 HU) at a known angle inside a 2D plane of a volume."""
    n = 81
    arr = np.full((n, n, n), -1000, np.int16)  # (Z, Y, X)
    t = np.linspace(-30, 30, 400)
    a = np.radians(angle_deg)
    u = np.round(40 + t * np.cos(a)).astype(int)  # first projected coordinate
    v = np.round(40 + t * np.sin(a)).astype(int)  # second projected coordinate
    for du in (-1, 0, 1):
        if plane == "YZ":    # proj = (y, z)
            arr[v, u + du, 40] = 2000
        elif plane == "XZ":  # proj = (x, z)
            arr[v, 40, u + du] = 2000
        else:                # "XY": proj = (x, y)
            arr[40, v, u + du] = 2000
    return sitk.GetImageFromArray(arr)


@pytest.mark.parametrize("plane", ["YZ", "XZ", "XY"])
@pytest.mark.parametrize("angle", [10.0, 35.0, 60.0])
def test_proyectar_plano_recovers_bar_angle(plane, angle):
    measured = proyectar_plano(_bar_volume(angle, plane), plano=plane, hu_umbral=300)
    # A principal axis has no sign: theta and theta +/- 180 describe the same line.
    diff = (measured - angle + 90.0) % 180.0 - 90.0
    assert abs(diff) < 2.0


def test_proyectar_plano_rejects_unknown_plane():
    with pytest.raises(ValueError):
        proyectar_plano(_bar_volume(10.0, "YZ"), plano="AB")


# ------------------------------------------------------------ DICOM + I/O
def test_find_dicom_series_dir_searches_nested_folders(dicom_patient):
    found = find_dicom_series_dir(dicom_patient, max_depth=2)
    assert found is not None
    assert os.path.normpath(found).endswith(os.path.join("study", "series"))


def test_find_dicom_series_dir_respects_max_depth(dicom_patient):
    assert find_dicom_series_dir(dicom_patient, max_depth=1) is None


def test_find_dicom_series_dir_returns_none_for_empty_folder(tmp_path):
    assert find_dicom_series_dir(str(tmp_path)) is None


def test_cargar_volumen_estandar_reads_series_in_rai(dicom_patient):
    vol = cargar_volumen_estandar(find_dicom_series_dir(dicom_patient))
    assert vol.GetSize()[2] == 6
    np.testing.assert_allclose(vol.GetSpacing(), (0.5, 0.7, 2.0))
    orient = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(vol.GetDirection())
    assert orient == "RAI"


def test_cargar_volumen_estandar_raises_without_series(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_volumen_estandar(str(tmp_path))


def test_guardar_volumen_como_nii_roundtrip(tmp_path):
    img = _volume()
    out_dir = tmp_path / "new" / "folder"  # does not exist yet
    guardar_volumen_como_nii(img, str(out_dir), "P001")
    path = out_dir / "Volumen_reorientado_paciente_P001.nii.gz"
    assert path.exists()
    back = sitk.ReadImage(str(path))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(img))
    np.testing.assert_allclose(back.GetSpacing(), img.GetSpacing())


def test_main_reorientation_on_synthetic_series(dicom_patient, tmp_path):
    out_dir = tmp_path / "out"
    series = find_dicom_series_dir(dicom_patient)
    vol = Etapa1.MainReorientation(series, str(out_dir), show=False)
    assert isinstance(vol, sitk.Image)
    written = list(out_dir.glob("Volumen_reorientado_paciente_*.nii.gz"))
    assert len(written) == 1
