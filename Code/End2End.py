"""
tmt_pipeline.py
===============
Pipeline end-to-end para la medición automática del músculo temporal (TMT)
a partir de una carpeta DICOM.

Flujo interno
-------------
1. Reorientación craneal        (reorientacion.py  → MainReorientation)
2. Extracción del corte axial   (corte_axial_optimo.py → MainOptAxialSingle)
3. Medición TMT                 (medicion_tmt.py   → Processing)

Interfaz pública
----------------
    MainTMT_Single(ruta_dicom, out_csv)
    MainTMT_Batch(root_dicoms,  out_csv)

Uso
---
    from tmt_pipeline import MainTMT_Single, MainTMT_Batch

    # Un solo paciente
    fila = MainTMT_Single(
        ruta_dicom = r"C:/datos/paciente_0",
        out_csv    = r"C:/resultados/tmt.csv"
    )

    # Varios pacientes (una subcarpeta por paciente)
    df = MainTMT_Batch(
        root_dicoms = r"C:/datos",
        out_csv     = r"C:/resultados/tmt_batch.csv"
    )

Dependencias
------------
    reorientacion.py, corte_axial_optimo.py, medicion_tmt.py
    (deben estar en el mismo directorio o en el PYTHONPATH)
"""

import os
import tempfile
import shutil
import glob
import pandas as pd
from datetime import datetime

# ── Módulos internos del pipeline ─────────────────────────────────────────────
from Etapa1      import MainReorientation, find_dicom_series_dir
from Etapa2 import MainOptAxialSingle
from Etapa3       import Processing


# =========================
# === Utilidades internas==
# =========================

def _procesar_paciente(ruta_dicom: str,
                       paciente_id: str,
                       tmp_dir: str) -> dict:
    """
    Ejecuta el pipeline completo para un paciente y retorna una fila de resultados.

    Orquesta los tres pasos del pipeline (reorientación, extracción del corte
    axial óptimo y medición TMT) sobre un directorio DICOM, usando una carpeta
    temporal para los archivos intermedios. No genera ningún archivo de salida
    permanente fuera del CSV final.

    Parámetros
    ----------
    ruta_dicom : str
        Ruta a la carpeta DICOM del paciente (debe contener una serie válida).
    paciente_id : str
        Identificador del paciente, usado como nombre en el CSV.
    tmp_dir : str
        Carpeta temporal donde se guardan el NIfTI reorientado y el corte axial.
        Se limpia automáticamente al finalizar el pipeline.

    Retorna
    -------
    dict
        Diccionario con las claves: scan, Thickness_R, Thickness_L,
        Mean Thickness, status, notas.
    """
    fila = {
        "scan":            paciente_id,
        "Thickness_R":     None,
        "Thickness_L":     None,
        "Mean Thickness":  None,
        "status":          "OK",
        "notas":           ""
    }

    # --- Paso 1: Reorientación ------------------------------------------------
    dir_nii    = os.path.join(tmp_dir, "nii")
    dir_cortes = os.path.join(tmp_dir, "cortes")
    os.makedirs(dir_nii,    exist_ok=True)
    os.makedirs(dir_cortes, exist_ok=True)

    MainReorientation(ruta_dicom, dir_nii, show=False)

    # Buscar el NIfTI generado
    nii_files = glob.glob(os.path.join(dir_nii, "*.nii.gz")) + \
                glob.glob(os.path.join(dir_nii, "*.nii"))

    if not nii_files:
        raise FileNotFoundError(
            f"Reorientación no generó ningún NIfTI en: {dir_nii}"
        )
    ruta_nii = nii_files[0]

    # --- Paso 2: Extracción del corte axial óptimo ----------------------------
    MainOptAxialSingle(
        ruta           = ruta_nii,
        out_dir_nii    = dir_cortes,
        mostrar_corte_final = False,
        metrics        = False
    )

    # Buscar el corte guardado
    corte_files = glob.glob(os.path.join(dir_cortes, "*.nii.gz")) + \
                  glob.glob(os.path.join(dir_cortes, "*.nii"))

    if not corte_files:
        raise FileNotFoundError(
            f"Extracción de corte no generó ningún NIfTI en: {dir_cortes}"
        )
    ruta_corte = corte_files[0]

    # --- Paso 3: Medición TMT -------------------------------------------------
    _, _, _, p1, p2, p1D, p2D, grosor_R, _, grosor_L, _ = Processing(ruta_corte)

    if grosor_R is None or grosor_L is None:
        raise ValueError("Processing no retornó mediciones válidas.")

    fila["Thickness_R"]    = round(float(grosor_R), 2)
    fila["Thickness_L"]    = round(float(grosor_L), 2)
    fila["Mean Thickness"] = round((float(grosor_R) + float(grosor_L)) / 2, 2)

    return fila


def _guardar_csv(filas: list, out_csv: str) -> None:
    """
    Guarda o actualiza el CSV de resultados con las filas proporcionadas.

    Si el archivo ya existe, agrega las filas nuevas al final sin duplicar
    la cabecera. Si no existe, lo crea desde cero.

    Parámetros
    ----------
    filas : list of dict
        Lista de filas a guardar, cada una con las claves del DataFrame.
    out_csv : str
        Ruta completa del archivo CSV de salida (incluye nombre y extensión).

    Retorna
    -------
    None
    """
    df_nuevo = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    if os.path.exists(out_csv):
        df_existente = pd.read_csv(out_csv)
        df_final     = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo

    df_final.to_csv(out_csv, index=False)


# =========================
# === SINGLE MAIN        ===
# =========================

def MainTMT_Single(ruta_dicom: str,
                   out_csv: str,
                   max_depth: int = 2) -> dict:
    """
    Ejecuta el pipeline TMT completo para un único paciente DICOM.

    Recibe una carpeta DICOM, aplica reorientación craneal, extrae el corte
    axial óptimo y mide el espesor del músculo temporal bilateral. Guarda
    el resultado en un CSV y lo retorna como diccionario.

    Los archivos intermedios (NIfTI reorientado, corte axial) se generan en
    una carpeta temporal y se eliminan automáticamente al finalizar.

    Parámetros
    ----------
    ruta_dicom : str
        Ruta a la carpeta DICOM del paciente. Puede ser la carpeta raíz del
        paciente o una carpeta que contenga la serie DICOM en subdirectorios
        (se busca automáticamente hasta max_depth niveles).
    out_csv : str
        Ruta completa del CSV de salida, incluyendo nombre y extensión .csv.
        Si el archivo ya existe, la nueva fila se agrega al final.
    max_depth : int, opcional
        Profundidad máxima para buscar la serie DICOM dentro de ruta_dicom.
        Por defecto 2.

    Retorna
    -------
    dict
        Fila de resultados con las claves: scan, Thickness_R, Thickness_L,
        Mean Thickness, status, notas.

    Ejemplo
    -------
        fila = MainTMT_Single(
            ruta_dicom = r"C:/datos/paciente_0",
            out_csv    = r"C:/resultados/tmt.csv"
        )
        print(fila["Mean Thickness"])
    """
    paciente_id = os.path.basename(os.path.normpath(ruta_dicom))
    print(f"\n[SINGLE] Procesando paciente: {paciente_id}")

    # Localizar la serie DICOM
    dicom_dir = find_dicom_series_dir(ruta_dicom, max_depth=max_depth)
    if dicom_dir is None:
        fila = {
            "scan": paciente_id, "Thickness_R": None,
            "Thickness_L": None, "Mean Thickness": None,
            "status": "ERROR", "notas": "No se encontró serie DICOM."
        }
        _guardar_csv([fila], out_csv)
        print(f"  [ERROR] No se encontró serie DICOM en: {ruta_dicom}")
        return fila

    tmp_dir = tempfile.mkdtemp(prefix=f"tmt_{paciente_id}_")
    try:
        fila = _procesar_paciente(dicom_dir, paciente_id, tmp_dir)
        print(f"  [OK] R={fila['Thickness_R']} mm | "
              f"L={fila['Thickness_L']} mm | "
              f"Media={fila['Mean Thickness']} mm")

    except Exception as e:
        fila = {
            "scan": paciente_id, "Thickness_R": None,
            "Thickness_L": None, "Mean Thickness": None,
            "status": "ERROR", "notas": str(e)
        }
        print(f"  [ERROR] {paciente_id} falló: {repr(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _guardar_csv([fila], out_csv)
    return fila


# =========================
# === BATCH MAIN         ===
# =========================

def MainTMT_Batch(root_dicoms: str,
                  out_csv: str,
                  max_depth: int = 2) -> pd.DataFrame:
    """
    Ejecuta el pipeline TMT completo en batch para múltiples pacientes DICOM.

    Itera sobre las subcarpetas de root_dicoms, trata cada una como un
    paciente independiente y llama a _procesar_paciente. Genera un único
    CSV consolidado con todos los resultados al finalizar.

    Estructura esperada de directorios
    -----------------------------------
    root_dicoms/
        0/   ← carpeta del paciente 0 (contiene la serie DICOM)
        1/
        2/
        ...

    Parámetros
    ----------
    root_dicoms : str
        Ruta a la carpeta raíz que contiene una subcarpeta por paciente.
    out_csv : str
        Ruta completa del CSV de salida, incluyendo nombre y extensión .csv.
        Si el archivo ya existe, los resultados se sobreescriben.
    max_depth : int, opcional
        Profundidad máxima para buscar la serie DICOM dentro de cada
        carpeta de paciente. Por defecto 2.

    Retorna
    -------
    pd.DataFrame
        DataFrame con una fila por paciente y columnas: scan, Thickness_R,
        Thickness_L, Mean Thickness, status, notas.

    Ejemplo
    -------
        df = MainTMT_Batch(
            root_dicoms = r"C:/datos",
            out_csv     = r"C:/resultados/tmt_batch.csv"
        )
        print(df)
    """
    root_dicoms = os.path.abspath(root_dicoms)

    patient_folders = sorted([
        os.path.join(root_dicoms, d)
        for d in os.listdir(root_dicoms)
        if os.path.isdir(os.path.join(root_dicoms, d))
    ], key=lambda p: os.path.basename(p))

    total = len(patient_folders)
    print(f"[BATCH] Pacientes detectados: {total}")
    print(f"[BATCH] Entrada: {root_dicoms}")
    print(f"[BATCH] Salida:  {out_csv}")

    filas  = []
    ok     = 0
    fail   = 0
    start  = datetime.now()

    for i, pdir in enumerate(patient_folders, start=1):
        paciente_id = os.path.basename(os.path.normpath(pdir))
        print(f"\n[{i}/{total}] Procesando paciente: {paciente_id}")

        dicom_dir = find_dicom_series_dir(pdir, max_depth=max_depth)
        if dicom_dir is None:
            fila = {
                "scan": paciente_id, "Thickness_R": None,
                "Thickness_L": None, "Mean Thickness": None,
                "status": "ERROR", "notas": "No se encontró serie DICOM."
            }
            print(f"  [ERROR] No se encontró serie DICOM en: {pdir}")
            filas.append(fila)
            fail += 1
            continue

        tmp_dir = tempfile.mkdtemp(prefix=f"tmt_{paciente_id}_")
        try:
            fila = _procesar_paciente(dicom_dir, paciente_id, tmp_dir)
            print(f"  [OK] R={fila['Thickness_R']} mm | "
                  f"L={fila['Thickness_L']} mm | "
                  f"Media={fila['Mean Thickness']} mm")
            ok += 1

        except Exception as e:
            fila = {
                "scan": paciente_id, "Thickness_R": None,
                "Thickness_L": None, "Mean Thickness": None,
                "status": "ERROR", "notas": str(e)
            }
            print(f"  [ERROR] {paciente_id} falló: {repr(e)}")
            fail += 1

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        filas.append(fila)

    # --- Guardar CSV consolidado ---
    df = pd.DataFrame(filas)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)

    duracion = datetime.now() - start
    hh, rem  = divmod(int(duracion.total_seconds()), 3600)
    mm, ss   = divmod(rem, 60)
    print(f"\n[BATCH] OK={ok} | FAIL={fail} | TOTAL={total} | "
          f"Duración: {hh:02d}:{mm:02d}:{ss:02d}")
    print(f"[BATCH] CSV guardado en: {out_csv}")

    return df