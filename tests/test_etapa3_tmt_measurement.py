"""Stage 3: temporalis muscle thickness measurement (Etapa3.py)."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import Etapa3
from Etapa3 import (
    calcular_grosor_musculo,
    crear_markup_line,
    exportar_TMT_a_markups,
    pixel_a_mundo,
    punto_mas_arriba_e_izquierda,
    punto_mas_arriba_y_derecha,
    reorient_to_ras,
    ventana,
)
from conftest import head_phantom_display, save_display_slice_as_nifti


# ------------------------------------------------------------- windowing
def test_ventana_maps_window_to_0_255():
    img = np.array([-1000.0, 0.0, 50.0, 100.0, 3000.0])
    out = ventana(img, nivel=50, ancho=100)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [0, 0, 127, 255, 255])


# ----------------------------------------------------------- orientation
def test_reorient_to_ras_flips_lps_axes():
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    lps = nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0]))  # L, P, S
    np.testing.assert_array_equal(reorient_to_ras(lps), data[::-1, ::-1, :])


def test_reorient_to_ras_adds_axis_to_2d_images():
    img = nib.Nifti1Image(np.ones((5, 6)), np.eye(4))
    assert reorient_to_ras(img).shape == (5, 6, 1)


def test_pixel_a_mundo_applies_affine():
    affine = np.array([[0.5, 0, 0, -100], [0, 0.5, 0, 20], [0, 0, 2.0, 7], [0, 0, 0, 1]])
    assert pixel_a_mundo(10, 4, 3, affine) == pytest.approx([-95.0, 22.0, 13.0])


# ----------------------------------------------------- thickness on a line
def _vertical_tangent():
    return (50, 20), (50, 180)  # (x, y) points of a vertical skull tangent


@pytest.mark.parametrize("width, spacing", [(20, (0.5, 0.5)), (11, (0.43, 0.43)), (30, (1.0, 1.0))])
def test_thickness_of_band_perpendicular_to_tangent(width, spacing):
    mask = np.zeros((200, 200), np.uint8)
    mask[:, 60 : 60 + width] = 1
    mm, line_pts = calcular_grosor_musculo(*_vertical_tangent(), mask, pixel_spacing=spacing)
    # Measured between the centres of the first and last muscle pixels.
    assert mm == pytest.approx((width - 1) * spacing[0])
    (x0, y0), (x1, y1) = line_pts
    assert y0 == y1  # perpendicular to a vertical tangent is horizontal
    assert {x0, x1} == {60, 60 + width - 1}


def test_thickness_uses_x_spacing_for_horizontal_measurements():
    mask = np.zeros((200, 200), np.uint8)
    mask[:, 60:80] = 1
    mm, _ = calcular_grosor_musculo(*_vertical_tangent(), mask, pixel_spacing=(0.5, 0.9))
    assert mm == pytest.approx(19 * 0.5)


def test_thickness_counts_only_first_contiguous_run():
    mask = np.zeros((200, 200), np.uint8)
    mask[:, 60:70] = 1   # 10 px
    mask[:, 75:95] = 1   # 20 px, separated by a 5 px gap
    mm, _ = calcular_grosor_musculo(*_vertical_tangent(), mask, pixel_spacing=(1.0, 1.0))
    assert mm in (pytest.approx(9.0), pytest.approx(19.0))  # one run, never 34 px across the gap


def test_thickness_without_muscle_is_zero():
    mm, line_pts = calcular_grosor_musculo(*_vertical_tangent(), np.zeros((200, 200), np.uint8), (0.5, 0.5))
    assert mm == 0.0 and line_pts is None


# ---------------------------------------------------- skull tangent points
@pytest.fixture
def disk_skull():
    yy, xx = np.mgrid[:300, :300]
    return (((xx - 150) ** 2 + (yy - 150) ** 2) <= 120 ** 2).astype(np.uint8)


def _on_mask(mask, p):
    return mask[int(p[1]), int(p[0])] == 1


def test_left_tangent_touches_left_edge_of_skull(disk_skull):
    p1, p2, candidates = punto_mas_arriba_e_izquierda(disk_skull)
    assert _on_mask(disk_skull, p1) and _on_mask(disk_skull, p2)
    assert p1[0] <= 35 and p2[0] <= 35          # leftmost column of the disk is 30
    assert p2[1] > p1[1]                         # second point is lower in the image
    assert len(candidates) > 0


def test_right_tangent_touches_right_edge_of_skull(disk_skull):
    p1, p2, _ = punto_mas_arriba_y_derecha(disk_skull)
    assert _on_mask(disk_skull, p1) and _on_mask(disk_skull, p2)
    assert p1[0] >= 265 and p2[0] >= 265        # rightmost column of the disk is 270
    assert p2[1] > p1[1]


def test_left_and_right_tangents_are_mirror_images(disk_skull):
    l1, l2, _ = punto_mas_arriba_e_izquierda(disk_skull)
    r1, r2, _ = punto_mas_arriba_y_derecha(disk_skull)
    assert l1[0] + r1[0] == pytest.approx(300, abs=2)
    assert l1[1] == r1[1]


# -------------------------------------------------------- Slicer markups
def test_crear_markup_line_structure():
    m = crear_markup_line("L", (10, 20), (30, 20), np.eye(4), slice_z=0)
    assert m["type"] == "Line" and m["coordinateSystem"] == "RAS"
    p0, p1 = m["controlPoints"]
    assert p0["position"] == [10, 20, 0] and p1["position"] == [30, 20, 0]
    assert p0["id"] != p1["id"]


def test_exportar_markups_writes_valid_json(tmp_path):
    nii = str(tmp_path / "s.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 1), np.float32), np.eye(4)), nii)
    out = tmp_path / "m.json"
    exportar_TMT_a_markups(nii, str(out), None, None, None, None, ((1, 2), (5, 2)), None)
    doc = json.loads(out.read_text())
    assert doc["@schema"].endswith("markups-schema-v1.0.0.json")
    assert [m["name"] for m in doc["markups"]] == ["Perpendicular_Derecha"]


# ------------------------------------------------- end-to-end on a phantom
@pytest.mark.parametrize("t_left, t_right", [(20, 30), (14, 14)])
def test_processing_measures_both_temporal_muscles(tmp_path, t_left, t_right):
    path = save_display_slice_as_nifti(
        head_phantom_display(t_left=t_left, t_right=t_right), str(tmp_path / "slice.nii.gz"), spacing=(0.5, 0.5)
    )
    (data, mask_bone, mask_muscle, p1, p2, p1D, p2D,
     mm_left, line_left, mm_right, line_right) = Etapa3.Processing(path)

    np.testing.assert_array_equal(data, head_phantom_display(t_left=t_left, t_right=t_right))
    # Only the two muscles survive the component filter (the brain is excluded).
    assert mask_muscle.sum() == 79 * (t_left + t_right)
    assert mm_left == pytest.approx((t_left - 1) * 0.5, abs=0.5)
    assert mm_right == pytest.approx((t_right - 1) * 0.5, abs=0.5)
    assert line_left is not None and line_right is not None


def test_main_tmt_measure_writes_csv(tmp_path):
    path = save_display_slice_as_nifti(head_phantom_display(), str(tmp_path / "Corte_axial_optimo_paciente_P1.nii.gz"))
    csv_dir = tmp_path / "results"
    csv_dir.mkdir()
    Etapa3.MainTMT_Measure(path, str(csv_dir))
    df = pd.read_csv(csv_dir / "resultados_temporales_P1.csv")
    row = df.iloc[0]
    assert row["scan"] == "P1"
    assert row["Thickness_R"] == pytest.approx(9.5, abs=0.5)    # 20 px phantom muscle
    assert row["Thickness_L"] == pytest.approx(14.5, abs=0.5)   # 30 px phantom muscle
    assert row["Mean Thickness"] == pytest.approx((row["Thickness_R"] + row["Thickness_L"]) / 2, abs=0.01)


def test_main_tmt_measure_reports_error_row_instead_of_crashing(tmp_path):
    bad = tmp_path / "Corte_axial_optimo_paciente_BAD.nii.gz"
    nib.save(nib.Nifti1Image(np.full((50, 50, 1), -1000, np.float32), np.eye(4)), str(bad))  # only air
    Etapa3.MainTMT_Measure(str(bad), str(tmp_path))
    df = pd.read_csv(tmp_path / "resultados_temporales_BAD.csv")
    assert df.iloc[0]["Thickness_R"] == "error"
