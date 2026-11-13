<div align="center">

<table width="100%"><tr>
<td align="left" valign="top">

# 🧠 Automatic TMT
### Automatic Measurement of Temporalis Muscle Thickness in Head CT Images

*A research-oriented pipeline for reproducible TMT measurement from cranial CT scans.*

![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-CT%20Scans-0077B6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Work%20In%20Progress-F4A261?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-8338EC?style=for-the-badge)
[![tests](https://github.com/Sebaskr6/auTMT/actions/workflows/tests.yml/badge.svg)](https://github.com/Sebaskr6/auTMT/actions/workflows/tests.yml)

</td>
<td align="right" valign="top" width="140">

<img width="60" alt="UIS logo" src="https://github.com/user-attachments/assets/c3706c58-b800-44cf-ba11-36ac2561635c" />
<br/>
<img width="60" alt="CT illustrative image" src="https://github.com/user-attachments/assets/ca85d70b-c582-49a9-9e3a-e1b21029385c" />

</td>
</tr></table>

</div>

---

## 📌 Overview

**Automatic TMT** supports clinical and research workflows through a **reproducible, automated pipeline** for measuring temporalis muscle thickness from head CT images. The project emphasizes robustness, transparency, and validation against expert annotations.

> 🏛️ Developed at **Universidad Industrial de Santander (UIS)** · Research Group: **Connectivity and Signal Processing (CPS)**

---

## 📂 Project Structure
```
auTMT/
├── Code/
│   ├── AuTMT.ipynb          # guided notebook (setup → stages → end-to-end)
│   ├── End2End.py           # DICOM → TMT in one run (single patient / batch)
│   ├── Etapa1.py            # Stage 1 · DICOM loading + PCA head reorientation
│   ├── Etapa2.py            # Stage 2 · eye detection (Hough) + optimal axial slice
│   └── Etapa3.py            # Stage 3 · HU segmentation + thickness in mm
├── tests/                   # pytest suite on synthetic phantoms
├── .github/workflows/       # CI: runs the tests on every push
├── requirements.txt
├── requirements-dev.txt     # requirements + pytest
└── 📄 README.md
```

---

## 🔁 Reproducibility Guide

| Step | Notebook | Description |
|------|----------|-------------|
| 1️⃣ | `01_environment_setup` | Install dependencies and configure the environment |
| 2️⃣ | `02_data_download` | Download and organize CT scan datasets |
| 3️⃣ | `03_preprocessing` | Image preprocessing and preliminary analysis |
| 4️⃣ | `04_measurement_pipeline` | TMT measurement execution by stages |
| 5️⃣ | `05_end_to_end` | Full pipeline run from input to results |

---

## 🧪 Tests

The pipeline is meant to run on real head CT scans, which cannot be shipped with the repository. The test suite therefore builds **small synthetic phantoms with known geometry** and checks that each stage recovers it:

| Stage | What is verified |
|-------|------------------|
| **1 · Reorientation** | rotation matrices are proper rotations and compose correctly; header rotation keeps voxels and rotates about the volume centre; PCA recovers the angle of a synthetic bone bar in every plane; DICOM series discovery, loading (RAI) and NIfTI export |
| **2 · Optimal slice** | mm ↔ % ↔ slice-index conversions and their edge cases (out of range, zero-width range, empty volumes); eye-pair selection against distractors; Hough eye detection on a two-eye phantom, including eyes that drift between slices |
| **3 · TMT measurement** | HU windowing, RAS reorientation, pixel → world coordinates; muscle thickness on bands of known width and spacing; skull tangent points; full `Processing` on a skull + brain + temporal-muscle phantom (expected thickness within 0.5 mm); CSV and 3D Slicer markup export |

```bash
pip install -r requirements-dev.txt
pytest
```

> Writing these tests uncovered a bug in Stage 2: detected eye centres were stored as `uint16`, so the tracker's `previous − current` subtraction wrapped around (`100 − 102 → 65534`) whenever a coordinate decreased between slices. With drifting eyes, estimated positions were off by up to ~30 px; they are now stored as `int32` and a regression test covers it.

---

## 👩‍💻 Authors & Advisor

<table>
<tr>
<td align="center"><b>Dana Cotes Cala</b><br/><sub>Undergraduate Student · UIS</sub></td>
<td align="center"><b>Kevin Rueda Muñoz</b><br/><sub>Undergraduate Student · UIS</sub></td>
<td align="center"><b>Said Pertuz</b><br/><sub>Advisor · CPS Research Group</sub></td>
</tr>
</table>

---

## ⚠️ Work in Progress

This repository is an **ongoing development effort**. The current code should be treated as a *progress snapshot*, not a finalized release. You may encounter:

- 🔧 Incomplete features
- 🐛 Bugs or unexpected behavior
- ✏️ Sections under active revision

---

## 📬 Contact

📩 **[automatictmt@gmail.com](mailto:automatictmt@gmail.com)**

---

## ⭐ Citation & Acknowledgment

If you use or build upon this work, please acknowledge the **Automatic TMT** project and **Universidad Industrial de Santander (UIS)**.

---

<div align="center"><i>Advancing reproducible medical image analysis through automation.</i></div>
