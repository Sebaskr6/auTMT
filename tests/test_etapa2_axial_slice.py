"""Stage 2: optimal axial slice selection via eye detection (Etapa2.py)."""

import os

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

import Etapa2
from Etapa2 import (
    _clamp,
    _k_from_y_orient,
    _map_y_index_itk,
    _mm_to_pct,
    _pct_to_k,
    _seleccionar_pareja_ojos,
    apply_threshold,
    detectar_ojos_por_consistencia,
    process_ct_volume_binary,
    resample_to_axial_LPS,
    save_axial_slice_as_nii,
    z_center_profile,
    z_of_plane_center_3d1slice,
)
from conftest import eye_phantom


# ------------------------------------------------------------ small helpers
@pytest.mark.parametrize("v, expected", [(5, 5), (-3, 0), (12, 10), (0, 0), (10, 10)])
def test_clamp(v, expected):
    assert _clamp(v, 0, 10) == expected


@pytest.mark.parametrize(
    "z, expected",
    [(50, 0.5), (0, 0.0), (100, 1.0), (150, 1.0), (-20, 0.0), (25.0, 0.25)],
)
def test_mm_to_pct(z, expected):
    assert _mm_to_pct(z, 0, 100) == pytest.approx(expected)


def test_mm_to_pct_handles_descending_range():
    # z_min > z_max (e.g. slices ordered superior -> inferior): still a valid ratio
    assert _mm_to_pct(75, 100, 0) == pytest.approx(0.25)


def test_mm_to_pct_zero_width_range_does_not_divide_by_zero():
    assert _mm_to_pct(5, 5, 5) == 0.0


@pytest.mark.parametrize("pct, sz, k", [(0.0, 101, 0), (1.0, 101, 100), (0.5, 101, 50), (0.3, 11, 3)])
def test_pct_to_k(pct, sz, k):
    assert _pct_to_k(pct, sz) == k


@pytest.mark.parametrize("pct", [-0.5, 1.7])
def test_pct_to_k_out_of_range_is_clamped(pct):
    assert 0 <= _pct_to_k(pct, 50) <= 49


@pytest.mark.parametrize("sz", [0, 1])
def test_pct_to_k_degenerate_volumes(sz):
    assert _pct_to_k(0.7, sz) == 0


@pytest.mark.parametrize("y, Z, k", [(0, 100, 99), (99, 100, 0), (49.6, 100, 49), (-5, 100, 99), (500, 100, 0)])
def test_k_from_y_orient_inverts_display_axis(y, Z, k):
    assert _k_from_y_orient(y, Z) == k


def test_map_y_index_itk_flips_and_clamps():
    img = sitk.Image([8, 10, 4], sitk.sitkInt16)
    assert _map_y_index_itk(img, 0) == 9
    assert _map_y_index_itk(img, 9) == 0
    assert _map_y_index_itk(img, 3) == 6
    assert _map_y_index_itk(img, -4) == 9   # clamped to 0 first
    assert _map_y_index_itk(img, 40) == 0   # clamped to 9 first


def test_apply_threshold_is_inclusive_and_uint8():
    vol = np.array([-21, -20, 0, 20, 21], dtype=float)
    mask = apply_threshold(vol, -20, 20)
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, [0, 1, 1, 1, 0])


def test_process_ct_volume_binary_from_nifti(tmp_path):
    data = np.full((10, 10, 10), -1000.0)
    data[3:6, 3:6, 3:6] = 10.0  # aqueous-humour-like density
    path = str(tmp_path / "v.nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    mask = process_ct_volume_binary(path)
    assert mask.sum() == 27


# ------------------------------------------------------- geometry in mm
def _oblique_image(size=(10, 12, 20), spacing=(0.7, 0.7, 2.5), origin=(1.0, -2.0, 30.0)):
    img = sitk.Image(list(size), sitk.sitkInt16)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


def test_resample_to_axial_lps_forces_identity_direction():
    img = _oblique_image()
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    img.SetDirection((c, -s, 0, s, c, 0, 0, 0, 1))
    out = resample_to_axial_LPS(img)
    np.testing.assert_allclose(out.GetDirection(), np.eye(3).ravel())
    assert out.GetSize() == img.GetSize()
    assert out.GetSpacing() == img.GetSpacing()
    assert out.GetOrigin() == img.GetOrigin()


def test_z_center_profile_is_linear_in_slice_index():
    img = _oblique_image()
    z = z_center_profile(img)
    assert z.shape == (20,)
    np.testing.assert_allclose(z, 30.0 + 2.5 * np.arange(20))


def test_z_of_plane_center_for_single_slice():
    img = _oblique_image(size=(10, 12, 1), origin=(0.0, 0.0, -47.5))
    assert z_of_plane_center_3d1slice(img) == pytest.approx(-47.5)


def test_save_axial_slice_as_nii(tmp_path):
    arr = np.arange(20 * 12 * 10, dtype=np.int16).reshape(20, 12, 10)  # (Z, Y, X)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.7, 2.5))
    img.SetOrigin((0.0, 0.0, 30.0))
    vol_path = str(tmp_path / "vol.nii.gz")
    sitk.WriteImage(img, vol_path)

    out_path, z_mm, pct = save_axial_slice_as_nii(
        vol_path, k=8, out_dir=str(tmp_path / "out"), paciente_id="P7", zc_range=(30.0, 30.0 + 2.5 * 19)
    )
    assert os.path.basename(out_path) == "Corte_axial_optimo_paciente_P7.nii.gz"
    sl = sitk.ReadImage(out_path)
    assert sl.GetSize() == (10, 12, 1)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(sl)[0], arr[8])
    assert z_mm == pytest.approx(30.0 + 2.5 * 8)
    assert pct == pytest.approx(8 / 19)


def test_save_axial_slice_clamps_k(tmp_path):
    img = sitk.GetImageFromArray(np.zeros((5, 4, 4), np.int16))
    vol_path = str(tmp_path / "vol.nii.gz")
    sitk.WriteImage(img, vol_path)
    out_path, z_mm, _ = save_axial_slice_as_nii(vol_path, 99, str(tmp_path), "X", (0.0, 4.0))
    assert z_mm == pytest.approx(4.0)  # last slice, not an IndexError


# ---------------------------------------------------- eye pair selection
def _track(x, y, n=6, jitter=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([x + rng.normal(0, jitter, n), y + rng.normal(0, jitter, n)])


def test_pair_selection_empty_and_single():
    assert _seleccionar_pareja_ojos([], (200, 200)) == (None, None)
    left, right = _seleccionar_pareja_ojos([_track(50, 100)], (200, 200))
    assert right is None
    np.testing.assert_allclose(left, [50, 100], atol=1)


def test_pair_selection_prefers_aligned_pair_at_plausible_distance():
    eyes = [_track(60, 140, seed=1), _track(140, 140, seed=2)]
    distractors = [
        _track(100, 60, n=6, seed=3),   # same persistence, misaligned in y
        _track(70, 141, n=6, seed=4),   # aligned but too close to the left eye
    ]
    left, right = _seleccionar_pareja_ojos(distractors[:1] + eyes + distractors[1:], (200, 200))
    np.testing.assert_allclose(left, [60, 140], atol=1)
    np.testing.assert_allclose(right, [140, 140], atol=1)


def test_pair_selection_returns_left_eye_first():
    left, right = _seleccionar_pareja_ojos([_track(150, 120), _track(40, 120)], (200, 200))
    assert left[0] < right[0]


# ------------------------------------------- full eye detection (Hough)
def _detect(vol, tmp_path):
    return detectar_ojos_por_consistencia(
        vol,
        "Volumen_reorientado_paciente_TEST.nii.gz",
        mostrar_corte_final=False,
        out_dir_mosaico=str(tmp_path / "mosaics"),
        out_dir_paciente=str(tmp_path / "patients"),
    )


def test_detects_both_eyes_on_aligned_phantom(tmp_path):
    left, right = _detect(eye_phantom(drift=0.0), tmp_path)
    assert left is not None and right is not None
    assert left[0] == pytest.approx(60, abs=3)
    assert right[0] == pytest.approx(140, abs=3)
    assert left[1] == pytest.approx(right[1], abs=2)  # same height


@pytest.mark.parametrize("drift", [-3.0, -1.5, 1.5, 3.0])
def test_detects_both_eyes_when_centres_drift_between_slices(drift, tmp_path):
    """Eye centres move a little from one coronal slice to the next.

    Detected centres were stored as uint16, so ``track[-1] - c`` wrapped
    around to ~65535 whenever a coordinate decreased (e.g. 100 - 102 -> 65534).
    Tracks were broken and the estimated eye positions drifted by up to ~30 px.
    """
    left, right = _detect(eye_phantom(drift=drift), tmp_path)
    assert left is not None and right is not None
    assert left[0] == pytest.approx(60, abs=5)
    assert right[0] == pytest.approx(140, abs=5)


def test_returns_none_when_there_are_no_eyes(tmp_path):
    assert _detect(np.zeros((200, 40, 200), np.uint8), tmp_path) == (None, None)


def test_optional_outputs_are_written(tmp_path):
    collector = []
    detectar_ojos_por_consistencia(
        eye_phantom(),
        "Volumen_reorientado_paciente_TEST.nii.gz",
        guardar_mosaico=True,
        collector=collector,
        mostrar_corte_final=True,
        corte_idx_visual=20,
        out_dir_mosaico=str(tmp_path / "mosaics"),
        out_dir_paciente=str(tmp_path / "patients"),
    )
    assert (tmp_path / "mosaics" / "Mosaico_imagenes_deteccion_ojos_TEST.png").exists()
    assert (tmp_path / "patients" / "Paciente_TEST_corte_final.png").exists()
    assert len(collector) == 1 and collector[0].ndim == 3
