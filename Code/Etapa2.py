import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import os
import math
import glob
from datetime import datetime
import pandas as pd


# =========================
# === Utilidades geom.  ===
# =========================

def resample_to_axial_LPS(img: sitk.Image,
                           interpolator=sitk.sitkLinear,
                           default_value=-3024) -> sitk.Image:
    """
    Remuestrea un volumen a orientación axial LPS con dirección identidad.

    Mantiene el mismo origen, spacing y tamaño del volumen original,
    forzando la matriz de dirección a la identidad (ejes alineados con LPS).
    Es el paso previo obligatorio antes de cualquier extracción de cortes.

    Parámetros
    ----------
    img : sitk.Image
        Volumen de entrada en cualquier orientación.
    interpolator : int, opcional
        Interpolador de SimpleITK. Por defecto sitk.sitkLinear.
    default_value : int o float, opcional
        Valor de relleno para vóxeles fuera del FOV original (HU).
        Por defecto -3024.

    Retorna
    -------
    sitk.Image
        Volumen remuestrado con dirección identidad (axial LPS).
    """
    axial_direction = (1, 0, 0,
                       0, 1, 0,
                       0, 0, 1)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(axial_direction)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetDefaultPixelValue(default_value)

    return resampler.Execute(img)


def _clamp(v, lo, hi):
    """
    Limita el valor 'v' al intervalo [lo, hi].

    Parámetros
    ----------
    v : numeric
        Valor a limitar.
    lo : numeric
        Límite inferior del intervalo.
    hi : numeric
        Límite superior del intervalo.

    Retorna
    -------
    numeric
        Valor limitado al rango [lo, hi].
    """
    return max(lo, min(hi, v))


def _map_y_index_itk(img: sitk.Image, y_index_ojos: int) -> int:
    """
    Mapea el índice Y del espacio de imagen orientado al índice Y de ITK (LPS).

    Dado que ITK almacena el eje Y en orden inverso respecto a la visualización
    estándar, esta función aplica la transformación sy - 1 - y.

    Parámetros
    ----------
    img : sitk.Image
        Volumen de referencia para obtener el tamaño en Y.
    y_index_ojos : int
        Índice Y en el espacio de imagen orientado (visualización).

    Retorna
    -------
    int
        Índice Y correspondiente en el espacio ITK (LPS).
    """
    sy = img.GetSize()[1]
    y = int(_clamp(y_index_ojos, 0, sy - 1))
    return sy - 1 - y


def _z_mm_fila(img: sitk.Image, j0: int, k: int) -> float:
    """
    Calcula la coordenada Z física (mm) de una fila axial en el plano coronal Y=j0.

    Promedia las coordenadas Z de los extremos izquierdo y derecho en X
    para reducir el efecto de inclinaciones residuales.

    Parámetros
    ----------
    img : sitk.Image
        Volumen de referencia.
    j0 : int
        Índice Y del plano coronal (espacio ITK).
    k : int
        Índice del corte axial.

    Retorna
    -------
    float
        Coordenada Z física (mm) promediada entre los extremos en X.
    """
    sx, _, _ = img.GetSize()
    z0 = img.TransformIndexToPhysicalPoint((0,    j0, k))[2]
    z1 = img.TransformIndexToPhysicalPoint((sx-1, j0, k))[2]
    return 0.5 * (float(z0) + float(z1))


def z_center_profile(img: sitk.Image) -> np.ndarray:
    """
    Calcula el perfil de coordenadas Z físicas en el centro del plano para todos los cortes.

    Evalúa la coordenada Z en el píxel central (cx, cy) de cada corte axial k,
    lo que reproduce exactamente el valor Z que muestra 3D Slicer en su visor.

    Parámetros
    ----------
    img : sitk.Image
        Volumen axial LPS de referencia.

    Retorna
    -------
    np.ndarray, shape (sz,)
        Array con la coordenada Z física (mm) del centro de cada corte axial.
    """
    sx, sy, sz = img.GetSize()
    cx, cy = sx // 2, sy // 2
    Zc = np.empty(sz, dtype=float)
    for k in range(sz):
        Zc[k] = float(img.TransformIndexToPhysicalPoint((cx, cy, k))[2])
    return Zc


def _mm_to_pct(z_mm: float, z_min: float, z_max: float) -> float:
    """
    Convierte una coordenada Z física (mm) a porcentaje normalizado del rango [z_min, z_max].

    Parámetros
    ----------
    z_mm : float
        Coordenada Z en mm a convertir.
    z_min : float
        Coordenada Z mínima del rango (0%).
    z_max : float
        Coordenada Z máxima del rango (100%).

    Retorna
    -------
    float
        Porcentaje en el rango [0.0, 1.0].
    """
    denom = (z_max - z_min) if (z_max != z_min) else 1.0
    return _clamp((float(z_mm) - float(z_min)) / denom, 0.0, 1.0)


def _pct_to_k(pct: float, sz: int) -> int:
    """
    Convierte un porcentaje normalizado [0..1] al índice de corte axial correspondiente.

    Parámetros
    ----------
    pct : float
        Porcentaje normalizado en [0.0, 1.0].
    sz : int
        Número total de cortes axiales en el volumen.

    Retorna
    -------
    int
        Índice de corte k en el rango [0, sz-1].
    """
    return int(_clamp(round(float(pct) * float(max(0, sz - 1))), 0, max(0, sz - 1)))


def _k_from_y_orient(y_orient: float, Z: int) -> int:
    """
    Convierte la coordenada Y del espacio de visualización al índice k axial.

    Aplica la inversa de la cadena de transformaciones (transpose + rot90CW + flip -1)
    usada en el pipeline de visualización.

    Parámetros
    ----------
    y_orient : float
        Coordenada Y en el espacio de visualización orientado.
    Z : int
        Número total de cortes axiales en el volumen.

    Retorna
    -------
    int
        Índice de corte k en el rango [0, Z-1].
    """
    return int(_clamp(round((Z - 1) - float(y_orient)), 0, Z - 1))


def z_of_plane_center_3d1slice(img3d_1sl: sitk.Image) -> float:
    """
    Obtiene la coordenada Z física (mm) del centro del plano para un volumen 3D con Z=1.

    Reproduce el valor Z que muestra 3D Slicer para un corte extraído
    como imagen 3D de un solo slice.

    Parámetros
    ----------
    img3d_1sl : sitk.Image
        Imagen 3D con exactamente un corte en la dimensión Z (GetSize()[2] == 1).

    Retorna
    -------
    float
        Coordenada Z física (mm) del píxel central del plano.
    """
    sx, sy, _ = img3d_1sl.GetSize()
    return float(img3d_1sl.TransformIndexToPhysicalPoint((sx // 2, sy // 2, 0))[2])


# =========================
# === Guardar corte     ===
# =========================

def save_axial_slice_as_nii(volume_path: str, k: int, out_dir: str,
                             paciente_id: str, zc_range: tuple,
                             z_label_mm: float = None) -> tuple:
    """
    Extrae el corte axial k de un volumen NIfTI y lo guarda como 3D (Z=1).

    Remuestrea el volumen a axial LPS antes de extraer el corte, calcula
    la coordenada Z física del centro del plano y devuelve información
    de posición para trazabilidad.

    Parámetros
    ----------
    volume_path : str
        Ruta al archivo NIfTI (.nii o .nii.gz) del volumen completo.
    k : int
        Índice del corte axial a extraer.
    out_dir : str
        Carpeta de salida donde se guardará el archivo .nii.gz.
    paciente_id : str
        Identificador del paciente, usado en el nombre del archivo.
    zc_range : tuple of (float, float)
        Rango (z_min, z_max) del perfil Z central, usado para calcular
        el porcentaje de posición del corte guardado.
    z_label_mm : float, opcional
        Coordenada Z en mm para incluir en el nombre del archivo como
        etiqueta informativa. Si es None, se usa z_center_mm_saved.

    Retorna
    -------
    tuple of (str, float, float)
        - Ruta absoluta del archivo .nii.gz guardado.
        - Coordenada Z física (mm) del centro del plano guardado.
        - Porcentaje de posición del corte guardado en el rango zc_range.
    """
    img = sitk.ReadImage(str(volume_path))
    img = resample_to_axial_LPS(img)
    sx, sy, sz = img.GetSize()
    k = int(max(0, min(k, sz - 1)))

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([sx, sy, 1])
    extractor.SetIndex([0, 0, k])
    slice_3d = extractor.Execute(img)

    z_center_mm_saved = z_of_plane_center_3d1slice(slice_3d)
    zc_min, zc_max = float(zc_range[0]), float(zc_range[1])
    pct_saved = _mm_to_pct(z_center_mm_saved, zc_min, zc_max)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"Corte_axial_optimo_paciente_{paciente_id}.nii.gz"
    out_path = os.path.join(out_dir, fname)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_path)
    writer.UseCompressionOn()
    writer.Execute(slice_3d)

    return out_path, z_center_mm_saved, pct_saved


# =========================
# === Núcleo "porcentaje"
# =========================

def inferir_corte_auto_mm_desde_ojos_pct(volume_path: str,
                                          corte_idx_ojos: int,
                                          y_ojo1: float,
                                          y_ojo2: float,
                                          delta_superior_mm: float = 20.4,
                                          refine_by_center_mm: bool = True):
    """
    Infiere el índice del corte axial óptimo a partir de la posición de los ojos.

    1) Localiza la línea de ojos en el volumen usando las coordenadas Y detectadas.
    2) Calcula z_auto_mm = z_eyes + delta_superior_mm como coordenada objetivo.
    3) Convierte z_auto_mm a porcentaje usando el rango del perfil Z_center(k).
    4) Obtiene k_auto_pct = round(f * (sz-1)).
    5) Si refine_by_center_mm, elige el k cuyo Z_center esté más cerca de z_auto_mm.

    Parámetros
    ----------
    volume_path : str
        Ruta al archivo NIfTI del volumen reorientado.
    corte_idx_ojos : int
        Índice del corte coronal donde se midieron los ojos.
    y_ojo1 : float
        Coordenada Y del primer ojo en el espacio de visualización.
    y_ojo2 : float
        Coordenada Y del segundo ojo en el espacio de visualización.
    delta_superior_mm : float, opcional
        Desplazamiento en mm hacia superior desde la línea de ojos
        para definir el corte objetivo. Por defecto 20.4.
    refine_by_center_mm : bool, opcional
        Si es True, refina k eligiendo el corte cuyo Z_center es el más
        cercano a z_auto_mm. Por defecto True.

    Retorna
    -------
    dict or None
        Diccionario con las claves: 'j0_itk', 'k_eye', 'k_auto_pct', 'k_auto',
        'z_eye_mm', 'z_auto_mm', 'z_center_min_mm', 'z_center_max_mm',
        'z_auto_mm_snap_center'. Retorna None si los ojos no están disponibles.
    """
    img = sitk.ReadImage(str(volume_path))
    sx, sy, sz = img.GetSize()

    if y_ojo1 is None or y_ojo2 is None:
        return None

    y_avg_orient = 0.5 * (float(y_ojo1) + float(y_ojo2))
    j0_itk = _map_y_index_itk(img, corte_idx_ojos)
    k_eye = _k_from_y_orient(y_avg_orient, sz)
    z_eye_mm = _z_mm_fila(img, j0_itk, k_eye)

    img = resample_to_axial_LPS(img)
    sx, sy, sz = img.GetSize()
    z_auto_mm = float(z_eye_mm) + float(delta_superior_mm)

    Zc = z_center_profile(img)
    zc_min, zc_max = float(Zc.min()), float(Zc.max())

    f = _mm_to_pct(z_auto_mm, zc_min, zc_max)
    k_auto_pct = _pct_to_k(f, sz)

    if refine_by_center_mm:
        k_auto = int(np.argmin(np.abs(Zc - z_auto_mm)))
    else:
        k_auto = int(k_auto_pct)

    return {
        "j0_itk":                j0_itk,
        "k_eye":                 k_eye,
        "k_auto_pct":            int(k_auto_pct),
        "k_auto":                k_auto,
        "z_eye_mm":              float(z_eye_mm),
        "z_auto_mm":             float(z_auto_mm),
        "z_center_min_mm":       float(zc_min),
        "z_center_max_mm":       float(zc_max),
        "z_auto_mm_snap_center": float(Zc[k_auto])
    }


# =========================
# === Carga / umbral    ===
# =========================

def load_ct_volume(path: str) -> np.ndarray:
    """
    Carga un volumen NIfTI y lo reorienta al espacio canónico RAS.

    Parámetros
    ----------
    path : str
        Ruta al archivo NIfTI (.nii o .nii.gz).

    Retorna
    -------
    np.ndarray
        Array 3D con los datos del volumen en orientación RAS canónica.
    """
    nii = nib.load(path)
    nii_ras = nib.as_closest_canonical(nii)
    return nii_ras.get_fdata()


def apply_threshold(volume: np.ndarray, lower: float = -20, upper: float = 20) -> np.ndarray:
    """
    Aplica un umbral de intensidad y devuelve una máscara binaria.

    Parámetros
    ----------
    volume : np.ndarray
        Array 3D del volumen CT.
    lower : float, opcional
        Límite inferior del umbral (HU). Por defecto -20.
    upper : float, opcional
        Límite superior del umbral (HU). Por defecto 20.

    Retorna
    -------
    np.ndarray, dtype uint8
        Máscara binaria (1 donde lower <= vóxel <= upper, 0 en otro caso).
    """
    mask = (volume >= lower) & (volume <= upper)
    return mask.astype(np.uint8)


def process_ct_volume_binary(path: str, threshold_low: float = -20, threshold_high: float = 20) -> np.ndarray:
    """
    Carga un volumen CT y devuelve su máscara binaria segmentada por umbral HU.

    Combina load_ct_volume y apply_threshold en un único paso.
    Se usa para aislar estructuras de densidad similar al humor acuoso
    del ojo (aproximadamente 0 HU) antes de la detección de ojos.

    Parámetros
    ----------
    path : str
        Ruta al archivo NIfTI del volumen CT.
    threshold_low : float, opcional
        Límite inferior del rango HU a segmentar. Por defecto -20.
    threshold_high : float, opcional
        Límite superior del rango HU a segmentar. Por defecto 20.

    Retorna
    -------
    np.ndarray, dtype uint8
        Máscara binaria 3D con las estructuras dentro del rango HU indicado.
    """
    vol = load_ct_volume(path)
    return apply_threshold(vol, threshold_low, threshold_high)


# =========================
# === Recolector Batch  ===
# =========================

def _gather_nii_files(input_paths_or_dir, recursive: bool = False) -> list:
    """
    Recopila todos los archivos NIfTI (.nii / .nii.gz) en las rutas indicadas.

    Acepta tanto una sola ruta (str) como una lista de rutas. Cada ruta puede
    ser un directorio o un archivo individual. Elimina duplicados y ordena
    el resultado alfabéticamente.

    Parámetros
    ----------
    input_paths_or_dir : str o list of str
        Ruta(s) de entrada: carpeta(s) o archivo(s) NIfTI individuales.
    recursive : bool, opcional
        Si es True, busca recursivamente en subdirectorios. Por defecto False.

    Retorna
    -------
    list of str
        Lista ordenada y sin duplicados de rutas absolutas a archivos NIfTI.
    """
    def _is_nifti(p):
        p_lower = p.lower()
        return p_lower.endswith(".nii") or p_lower.endswith(".nii.gz")

    files = []
    paths = input_paths_or_dir if isinstance(input_paths_or_dir, (list, tuple)) else [input_paths_or_dir]

    for p in paths:
        if os.path.isdir(p):
            pattern1 = os.path.join(p, "**", "*.nii")    if recursive else os.path.join(p, "*.nii")
            pattern2 = os.path.join(p, "**", "*.nii.gz") if recursive else os.path.join(p, "*.nii.gz")
            files.extend(glob.glob(pattern1, recursive=recursive))
            files.extend(glob.glob(pattern2, recursive=recursive))
        elif os.path.isfile(p) and _is_nifti(p):
            files.append(p)

    return sorted(list(dict.fromkeys(files)))


def _guardar_metricas(df: pd.DataFrame, metrics_path: str) -> None:
    """
    Guarda un DataFrame de métricas como archivo CSV.

    Crea los directorios intermedios si no existen y escribe el archivo
    sin índice de fila.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las métricas a guardar.
    metrics_path : str
        Ruta completa del archivo CSV de salida (incluye nombre y extensión).

    Retorna
    -------
    None
    """
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    df.to_csv(metrics_path, index=False)
    print(f"  [OK] Tabla de métricas guardada en: {metrics_path}")


# =========================
# === Detección de ojos ===
# =========================

def _seleccionar_pareja_ojos(tracks_lista: list, shape: tuple,
                              y_tol: float = 10,
                              dmin_frac: float = 0.18,
                              dmax_frac: float = 0.55):
    """
    Selecciona la pareja de trayectorias que mejor representa los dos ojos.

    Evalúa todas las combinaciones de pares de trayectorias y asigna un
    puntaje según persistencia, separación horizontal esperada entre ojos
    y alineación vertical. Retorna las posiciones medias del par óptimo.

    Parámetros
    ----------
    tracks_lista : list of np.ndarray
        Lista de trayectorias, cada una como array de forma (N, 2) con
        columnas (x, y).
    shape : tuple of (int, int)
        Dimensiones (H, W) del plano de detección.
    y_tol : float, opcional
        Tolerancia en píxeles de diferencia vertical entre los dos ojos.
        Por defecto 10.
    dmin_frac : float, opcional
        Fracción mínima del ancho W como separación horizontal válida.
        Por defecto 0.18.
    dmax_frac : float, opcional
        Fracción máxima del ancho W como separación horizontal válida.
        Por defecto 0.55.

    Retorna
    -------
    tuple of (np.ndarray or None, np.ndarray or None)
        Posiciones medias (x, y) del ojo izquierdo y derecho respectivamente.
        Retorna (None, None) si no se encuentra ninguna pareja válida.
    """
    def _stats(track_array):
        return {"x": float(track_array[:, 0].mean()),
                "y": float(track_array[:, 1].mean()),
                "n": int(len(track_array))}

    if len(tracks_lista) == 0:
        return None, None
    if len(tracks_lista) == 1:
        return tracks_lista[0].mean(axis=0)[:2], None

    H, W = shape[:2]
    dmin = dmin_frac * W
    dmax = dmax_frac * W
    stats = [_stats(t) for t in tracks_lista]

    best = None
    best_score = -1e9

    for i in range(len(tracks_lista)):
        for j in range(i + 1, len(tracks_lista)):
            ti, tj = tracks_lista[i], tracks_lista[j]
            si, sj = stats[i], stats[j]

            if si["x"] <= sj["x"]:
                L, R, tL, tR = si, sj, ti, tj
            else:
                L, R, tL, tR = sj, si, tj, ti

            dy = abs(L["y"] - R["y"])
            dx = R["x"] - L["x"]
            if dx <= 0:
                continue

            overlap = min(tL.shape[0], tR.shape[0])
            in_range_dx = (dmin <= dx <= dmax)
            y_penalty = max(0.0, dy - y_tol)

            score  = 4.0  * overlap
            score += 15.0 * (1.0 if in_range_dx else -1.0)
            score += 20.0 * (1.0 / (1.0 + dy))
            score += -5.0 * y_penalty

            if score > best_score:
                best_score = score
                best = (tL.mean(axis=0)[:2], tR.mean(axis=0)[:2])

    if best is None:
        return None, None
    return best[0], best[1]


def detectar_ojos_por_consistencia(volumen: np.ndarray,
                                    ruta: str,
                                    radio_min: int = 15,
                                    radio_max: int = 28,
                                    paso_radio: int = 2,
                                    tolerancia: int = 8,
                                    min_persistencia: int = 4,
                                    guardar_mosaico: bool = False,
                                    y_min_frac: float = 0.2,
                                    collector: list = None,
                                    mostrar_corte_final: bool = True,
                                    corte_idx_visual: int = 389,
                                    out_dir_mosaico: str = "Img_auto_mosaicos",
                                    out_dir_paciente: str = "Img_auto_paciente"):
    """
    Detecta los globos oculares en un volumen CT mediante Hough Circle Transform.

    Itera sobre los cortes coronales del volumen binario, aplica la transformada
    de Hough para detectar círculos (globos oculares), construye trayectorias
    de centros consistentes entre cortes y selecciona la pareja más plausible.

    Parámetros
    ----------
    volumen : np.ndarray
        Máscara binaria 3D del volumen CT, shape (X, Y, Z) o similar.
    ruta : str
        Ruta al archivo NIfTI original, usada para nombrar las salidas.
    radio_min : int, opcional
        Radio mínimo (px) para la búsqueda de círculos. Por defecto 15.
    radio_max : int, opcional
        Radio máximo (px) para la búsqueda de círculos. Por defecto 28.
    paso_radio : int, opcional
        Paso entre radios evaluados (no usado directamente en HoughCircles,
        conservado para compatibilidad). Por defecto 2.
    tolerancia : int, opcional
        Distancia máxima (px) para asociar un centro nuevo a una trayectoria
        existente en el tracking. Por defecto 8.
    min_persistencia : int, opcional
        Número mínimo de cortes en los que debe aparecer un círculo para
        considerarlo una detección válida. Por defecto 4.
    guardar_mosaico : bool, opcional
        Si es True, guarda un mosaico PNG con los cortes donde se detectaron
        círculos y sus contornos. Por defecto False.
    y_min_frac : float, opcional
        Fracción mínima de la altura H por debajo de la cual se descartan
        trayectorias (filtro para eliminar detecciones en el cráneo). Por
        defecto 0.2.
    collector : list o None, opcional
        Lista donde se acumula la imagen del corte final anotado, usada
        para construir el mosaico global en el batch. Por defecto None.
    mostrar_corte_final : bool, opcional
        Si es True, muestra y guarda el corte coronal 'corte_idx_visual'
        con los ojos detectados anotados. Por defecto True.
    corte_idx_visual : int, opcional
        Índice del corte coronal usado para la visualización final.
        Por defecto 389.
    out_dir_mosaico : str, opcional
        Carpeta de salida para los mosaicos por paciente. Por defecto
        "Img_auto_mosaicos".
    out_dir_paciente : str, opcional
        Carpeta de salida para las imágenes individuales anotadas.
        Por defecto "Img_auto_paciente".

    Retorna
    -------
    tuple of (tuple or None, tuple or None)
        Coordenadas (x, y) del ojo izquierdo y del ojo derecho
        respectivamente. Retorna (None, None) si no se detectó ninguna
        pareja válida.
    """
    os.makedirs(out_dir_mosaico, exist_ok=True)
    os.makedirs(out_dir_paciente, exist_ok=True)

    nombre_archivo = os.path.basename(ruta).split(".")[0].removeprefix("Volumen_reorientado_paciente_")
    ruta_salida_mosaico  = os.path.join(out_dir_mosaico,  f"Mosaico_imagenes_deteccion_ojos_{nombre_archivo}.png")
    ruta_salida_paciente = os.path.join(out_dir_paciente, f"Paciente_{nombre_archivo}_corte_final.png")

    vol_np = volumen
    num_cortes = vol_np.shape[1]
    centros_por_corte = []
    imagenes_detectadas = []

    # --- Detección por corte coronal ---
    for i in range(num_cortes):
        img = vol_np[:, i, :].astype(np.float32)
        H, W = img.shape[:2]
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_norm = cv2.rotate(img_norm, cv2.ROTATE_90_CLOCKWISE)
        img_norm = cv2.flip(img_norm, -1)
        img_blur = cv2.medianBlur(img_norm, 5)

        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(0.25 * W),
            param1=100,
            param2=16,
            minRadius=radio_min,
            maxRadius=radio_max
        )

        if circles is not None:
            centers = np.uint16(np.around(circles[0, :, :2]))
            centros_por_corte.append(centers)

            if guardar_mosaico:
                img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
                for (x, y, r) in np.uint16(np.around(circles[0, :])):
                    cv2.circle(img_color, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(img_color, (x, y), 2, (0, 0, 255), 3)
                imagenes_detectadas.append((i, img_color))
        else:
            centros_por_corte.append([])

    # --- Tracking simple por proximidad ---
    tracks = []
    for centers in centros_por_corte:
        for c in centers:
            agregado = False
            for track in tracks:
                if np.linalg.norm(np.array(track[-1]) - c) < tolerancia:
                    track.append(c)
                    agregado = True
                    break
            if not agregado:
                tracks.append([c])

    # --- Agrupación de trayectorias cercanas ---
    centros = [np.mean(np.array(t), axis=0) for t in tracks]
    agrupados = []
    for i, c in enumerate(centros):
        agregado = False
        for grupo in agrupados:
            if np.linalg.norm(grupo['centro'] - c) < 10:
                grupo['tracks'].append(tracks[i])
                grupo['centro'] = np.mean(
                    [np.mean(np.array(t), axis=0) for t in grupo['tracks']], axis=0
                )
                agregado = True
                break
        if not agregado:
            agrupados.append({'centro': c, 'tracks': [tracks[i]]})

    tracks = [np.vstack(grupo['tracks']) for grupo in agrupados]
    tracks = [t for t in tracks if len(t) >= min_persistencia]

    # --- Filtro vertical: descartar trayectorias demasiado altas ---
    H = vol_np.shape[0]
    if y_min_frac is not None:
        y_floor = float(y_min_frac) * float(H)
        tracks = [t for t in tracks if float(np.mean(t[:, 1])) >= y_floor]

    if len(tracks) == 0:
        print("  [WARN] No se detectó ninguna trayectoria válida.")
        return None, None

    H, W = vol_np.shape[0], vol_np.shape[2]
    ojo1, ojo2 = _seleccionar_pareja_ojos(tracks, (H, W))

    if ojo1 is None and ojo2 is None:
        print("  [WARN] No se encontró pareja plausible de ojos tras el filtro.")
        return None, None

    # --- Mosaico por paciente (opcional) ---
    if guardar_mosaico and len(imagenes_detectadas) > 0:
        n = len(imagenes_detectadas)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        h, w, _ = imagenes_detectadas[0][1].shape
        mosaico = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 30

        for idx, (i, img_color) in enumerate(imagenes_detectadas):
            r, c = idx // cols, idx % cols
            y0, y1 = r * h, (r + 1) * h
            x0, x1 = c * w, (c + 1) * w
            mosaico[y0:y1, x0:x1] = img_color
            cv2.putText(mosaico, f"Corte {i}", (x0 + 5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(ruta_salida_mosaico, mosaico)

    # --- Preparar imagen del corte coronal anotado ---
    corte = int(_clamp(corte_idx_visual, 0, vol_np.shape[1] - 1))
    img_raw = vol_np[:, corte, :].astype(np.float32)
    img_norm = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_norm = cv2.rotate(img_norm, cv2.ROTATE_90_CLOCKWISE)
    img_norm = cv2.flip(img_norm, -1)
    img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

    if ojo1 is not None:
        cv2.circle(img_color, tuple(np.int32(ojo1)), 5, (255, 0, 0), -1)
        cv2.putText(img_color, "Eye 1", tuple(np.int32(ojo1 + [5, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if ojo2 is not None:
        cv2.circle(img_color, tuple(np.int32(ojo2)), 5, (0, 255, 255), -1)
        cv2.putText(img_color, "Eye 2", tuple(np.int32(ojo2 + [5, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if collector is not None:
        collector.append(img_color.copy())

    # --- Corte final: mostrar y guardar (opcional) ---
    if mostrar_corte_final:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        cv2.imwrite(ruta_salida_paciente, img_color)

    return (tuple(ojo1) if ojo1 is not None else None,
            tuple(ojo2) if ojo2 is not None else None)


def MainOptAxial(ruta: str,
                 collector: list = None,
                 guardar_mosaico: bool = False,
                 y_min_frac: float = 0.2,
                 mostrar_corte_final: bool = True,
                 corte_idx_visual: int = 389,
                 out_dir_mosaico: str = "Img_auto_mosaicos",
                 out_dir_paciente: str = "Img_auto_paciente"):
    """
    Detecta los ojos en un volumen NIfTI y retorna sus coordenadas.

    Binariza el volumen CT usando un umbral HU propio del humor acuoso
    (-20 a 20 HU) y delega la detección en detectar_ojos_por_consistencia.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo NIfTI (.nii o .nii.gz) del volumen reorientado.
    collector : list o None, opcional
        Lista donde se acumula el tile del corte final anotado para el
        mosaico global del batch. Por defecto None.
    guardar_mosaico : bool, opcional
        Si es True, guarda el mosaico de cortes con detecciones por
        paciente. Por defecto False.
    y_min_frac : float, opcional
        Fracción mínima de altura para filtrar trayectorias. Por defecto 0.2.
    mostrar_corte_final : bool, opcional
        Si es True, muestra el corte coronal visual con ojos anotados.
        Por defecto True.
    corte_idx_visual : int, opcional
        Índice del corte coronal usado para la visualización. Por defecto 389.
    out_dir_mosaico : str, opcional
        Carpeta de salida para mosaicos por paciente. Por defecto
        "Img_auto_mosaicos".
    out_dir_paciente : str, opcional
        Carpeta de salida para cortes individuales anotados. Por defecto
        "Img_auto_paciente".

    Retorna
    -------
    tuple of (tuple or None, tuple or None)
        Coordenadas (x, y) del ojo izquierdo y derecho respectivamente.
    """
    mask = process_ct_volume_binary(ruta, threshold_low=-20, threshold_high=20)
    ojo_izq, ojo_der = detectar_ojos_por_consistencia(
        mask, ruta,
        guardar_mosaico=guardar_mosaico,
        y_min_frac=y_min_frac,
        collector=collector,
        mostrar_corte_final=mostrar_corte_final,
        corte_idx_visual=corte_idx_visual,
        out_dir_mosaico=out_dir_mosaico,
        out_dir_paciente=out_dir_paciente
    )
    return ojo_izq, ojo_der


# =========================
# === MAIN Individual   ===
# =========================

def MainOptAxialSingle(ruta: str,
                       corte_idx_visual: int = 389,
                       delta_superior_mm: float = 20.4,
                       out_dir_nii: str = "Cortes_auto",
                       guardar_mosaico: bool = False,
                       mostrar_corte_final: bool = True,
                       metrics: bool = False,
                       metrics_path: str = None):
    """
    Detecta el corte axial óptimo de un único volumen y lo guarda como NIfTI.

    Ejecuta el pipeline completo para un solo paciente:
    detección de ojos → inferencia del corte por porcentaje → guardado.
    Si metrics=True, guarda adicionalmente un CSV con los resultados
    del procesamiento en metrics_path.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo NIfTI (.nii o .nii.gz) del volumen reorientado.
    corte_idx_visual : int, opcional
        Índice del corte coronal donde se miden las coordenadas Y de los
        ojos. Por defecto 389.
    delta_superior_mm : float, opcional
        Desplazamiento en mm hacia superior desde la línea de ojos.
        Por defecto 20.4.
    out_dir_nii : str, opcional
        Carpeta de salida para el corte axial guardado. Por defecto
        "Cortes_auto".
    guardar_mosaico : bool, opcional
        Si es True, guarda el mosaico de detecciones por corte. Por
        defecto False.
    mostrar_corte_final : bool, opcional
        Si es True, muestra el corte coronal anotado con los ojos
        detectados. Por defecto True.
    metrics : bool, opcional
        Si es True, guarda un CSV con las métricas del procesamiento.
        Requiere que metrics_path esté definido. Por defecto False.
    metrics_path : str o None, opcional
        Ruta completa del CSV de métricas. Obligatorio si metrics=True.

    Retorna
    -------
    None

    Lanza
    -----
    ValueError
        Si metrics=True y metrics_path es None.
    """
    if metrics and metrics_path is None:
        raise ValueError("Debes definir metrics_path si metrics=True.")

    nombre_archivo = os.path.basename(ruta)
    paciente_id = os.path.splitext(os.path.splitext(nombre_archivo)[0])[0].removeprefix(
        "Volumen_reorientado_paciente_"
    )

    print(f"\n[SINGLE] Procesando paciente: {paciente_id}")

    _COLS = [
        "archivo", "paciente_id",
        "ojo_izq_x", "ojo_izq_y", "ojo_der_x", "ojo_der_y",
        "z_eye_mm", "z_auto_mm",
        "z_center_min_mm", "z_center_max_mm",
        "k_eye", "k_auto_pct", "k_auto",
        "pct_auto_target", "pct_auto_snap_center",
        "z_auto_mm_snap_center",
        "nii_auto_path", "z_saved_center_mm", "pct_saved_center",
        "status", "notas"
    ]

    fila = {col: None for col in _COLS}
    fila.update({"archivo": ruta, "paciente_id": paciente_id,
                 "status": "OK", "notas": ""})

    try:
        # === 1. Detectar ojos
        ojo_izq, ojo_der = MainOptAxial(
            ruta,
            collector=None,
            guardar_mosaico=guardar_mosaico,
            mostrar_corte_final=mostrar_corte_final,
            corte_idx_visual=corte_idx_visual
        )

        if ojo_izq is not None:
            fila["ojo_izq_x"] = float(ojo_izq[0])
            fila["ojo_izq_y"] = float(ojo_izq[1])
        else:
            fila["notas"] += "Sin ojo_izq; "

        if ojo_der is not None:
            fila["ojo_der_x"] = float(ojo_der[0])
            fila["ojo_der_y"] = float(ojo_der[1])
        else:
            fila["notas"] += "Sin ojo_der; "

        if ojo_izq is None or ojo_der is None:
            fila["status"] = "WARN"
            if metrics:
                _guardar_metricas(pd.DataFrame([fila], columns=_COLS), metrics_path)
            return

        # === 2. Inferir corte automático
        rsel = inferir_corte_auto_mm_desde_ojos_pct(
            volume_path=ruta,
            corte_idx_ojos=corte_idx_visual,
            y_ojo1=fila["ojo_izq_y"],
            y_ojo2=fila["ojo_der_y"],
            delta_superior_mm=delta_superior_mm
        )

        if rsel is None:
            fila["status"] = "WARN"
            fila["notas"] += "No se pudo inferir Z desde ojos; "
            if metrics:
                _guardar_metricas(pd.DataFrame([fila], columns=_COLS), metrics_path)
            return

        fila.update({
            "k_eye":                 rsel["k_eye"],
            "k_auto_pct":            rsel["k_auto_pct"],
            "k_auto":                rsel["k_auto"],
            "z_eye_mm":              rsel["z_eye_mm"],
            "z_auto_mm":             rsel["z_auto_mm"],
            "z_center_min_mm":       rsel["z_center_min_mm"],
            "z_center_max_mm":       rsel["z_center_max_mm"],
            "z_auto_mm_snap_center": rsel["z_auto_mm_snap_center"],
        })

        # === 3. Perfil Z central
        img_itk = sitk.ReadImage(str(ruta))
        img_itk = resample_to_axial_LPS(img_itk)
        Zc = z_center_profile(img_itk)
        zc_min, zc_max = float(Zc.min()), float(Zc.max())

        fila["pct_auto_target"]      = float(_mm_to_pct(fila["z_auto_mm"], zc_min, zc_max))
        fila["pct_auto_snap_center"] = float(_mm_to_pct(fila["z_auto_mm_snap_center"], zc_min, zc_max))

        # === 4. Guardar corte axial
        nii_path, zc_saved, pct_saved = save_axial_slice_as_nii(
            volume_path=ruta,
            k=int(fila["k_auto"]),
            out_dir=out_dir_nii,
            paciente_id=paciente_id,
            zc_range=(zc_min, zc_max),
            z_label_mm=fila["z_auto_mm_snap_center"]
        )

        fila["nii_auto_path"]     = nii_path
        fila["z_saved_center_mm"] = float(zc_saved)
        fila["pct_saved_center"]  = float(pct_saved)

        print(f"  [OK] Paciente {paciente_id} procesado. Corte guardado en: {nii_path}")

    except Exception as e:
        fila["status"] = "ERROR"
        fila["notas"]  = str(e)
        print(f"  [ERROR] Paciente {paciente_id} falló: {repr(e)}")

    if metrics:
        _guardar_metricas(pd.DataFrame([fila], columns=_COLS), metrics_path)


# =========================
# === MAIN Batch        ===
# =========================

def MainOptAxialBatch(input_paths_or_dir,
                      recursive: bool = False,
                      corte_idx_visual: int = 389,
                      delta_superior_mm: float = 20.4,
                      out_dir_nii: str = "Cortes_auto",
                      out_dir_mosaicos: str = "Img_auto_mosaicos",
                      out_dir_paciente: str = "Img_auto_paciente",
                      guardar_mosaico_por_paciente: bool = False,
                      mostrar_corte_final: bool = False,
                      construir_mosaico_final: bool = False,
                      mostrar_mosaico_final: bool = False,
                      usar_collector_en_batch: bool = True,
                      target_pct_override: float = None,
                      refine_by_center_mm: bool = True,
                      metrics: bool = False,
                      metrics_path: str = None) -> None:
    """
    Ejecuta el pipeline de extracción de corte axial óptimo para múltiples pacientes.

    Para cada volumen NIfTI encontrado:
    1) Detecta los ojos (o usa target_pct_override si se indica).
    2) Calcula z_auto_mm = z_eyes + delta y lo convierte a porcentaje
       del rango Zc, obteniendo k por porcentaje y refinando por mínimo
       error en Zc si refine_by_center_mm es True.
    3) Guarda el corte axial óptimo como .nii.gz en out_dir_nii.
    4) Si metrics=True, guarda adicionalmente un CSV con los resultados
       de todos los pacientes en metrics_path.

    Parámetros
    ----------
    input_paths_or_dir : str o list of str
        Carpeta raíz con los volúmenes NIfTI o lista de rutas individuales.
    recursive : bool, opcional
        Si es True, busca NIfTI recursivamente en subdirectorios. Por defecto False.
    corte_idx_visual : int, opcional
        Índice del corte coronal donde se miden las coordenadas Y de los
        ojos. Por defecto 389.
    delta_superior_mm : float, opcional
        Desplazamiento en mm hacia superior desde la línea de ojos.
        Por defecto 20.4.
    out_dir_nii : str, opcional
        Carpeta de salida para los cortes axiales guardados. Por defecto
        "Cortes_auto".
    out_dir_mosaicos : str, opcional
        Carpeta de salida para los mosaicos por paciente y el mosaico
        global. Por defecto "Img_auto_mosaicos".
    out_dir_paciente : str, opcional
        Carpeta de salida para los cortes individuales anotados. Por
        defecto "Img_auto_paciente".
    guardar_mosaico_por_paciente : bool, opcional
        Si es True, guarda el mosaico de detecciones por cada paciente.
        Por defecto False.
    mostrar_corte_final : bool, opcional
        Si es True, muestra el corte coronal anotado por cada paciente.
        Por defecto False.
    construir_mosaico_final : bool, opcional
        Si es True, construye y guarda un mosaico global con todos los
        pacientes al finalizar. Por defecto False.
    mostrar_mosaico_final : bool, opcional
        Si es True, muestra el mosaico global al finalizar el batch.
        Por defecto False.
    usar_collector_en_batch : bool, opcional
        Si es True, acumula los tiles de cada paciente para el mosaico
        global. Por defecto True.
    target_pct_override : float o None, opcional
        Si se indica (valor en [0..1]), ignora la detección de ojos y usa
        directamente este porcentaje para todos los pacientes. Por defecto None.
    refine_by_center_mm : bool, opcional
        Si es True, refina k eligiendo el corte cuyo Z_center es el más
        cercano al objetivo en mm. Por defecto True.
    metrics : bool, opcional
        Si es True, guarda el CSV de resultados en metrics_path al finalizar.
        Requiere que metrics_path esté definido. Por defecto False.
    metrics_path : str o None, opcional
        Ruta completa del CSV de métricas. Obligatorio si metrics=True.

    Retorna
    -------
    None

    Lanza
    -----
    ValueError
        Si metrics=True y metrics_path es None.
    """
    if metrics and metrics_path is None:
        raise ValueError("Debes definir metrics_path si metrics=True.")

    _COLS = [
        "archivo", "paciente_id",
        "ojo_izq_x", "ojo_izq_y", "ojo_der_x", "ojo_der_y",
        "z_eye_mm", "z_auto_mm",
        "z_center_min_mm", "z_center_max_mm",
        "k_eye", "k_auto_pct", "k_auto",
        "pct_auto_target", "pct_auto_snap_center",
        "z_auto_mm_snap_center",
        "nii_auto_path", "z_saved_center_mm", "pct_saved_center",
        "status", "notas"
    ]

    files = _gather_nii_files(input_paths_or_dir, recursive=recursive)
    if not files:
        print("[BATCH] No se encontraron volúmenes NIfTI para procesar.")
        return

    total = len(files)
    print(f"[BATCH] Volúmenes detectados: {total}")
    print(f"[BATCH] Entrada: {input_paths_or_dir}")
    print(f"[BATCH] Salida:  {out_dir_nii}")

    # Solo se construye la lista de registros si metrics está activo
    registros = [] if metrics else None
    tiles_finales = [] if usar_collector_en_batch else None
    start = datetime.now()

    for idx, ruta in enumerate(files, 1):
        nombre_archivo = os.path.basename(ruta)
        paciente_id = os.path.splitext(os.path.splitext(nombre_archivo)[0])[0].removeprefix(
            "Volumen_reorientado_paciente_"
        )

        print(f"[{idx}/{total}] Procesando paciente: {paciente_id}")

        fila = {col: None for col in _COLS}
        fila.update({"archivo": ruta, "paciente_id": paciente_id,
                     "status": "OK", "notas": ""})

        try:
            # === 1. Detectar ojos
            if target_pct_override is None:
                ojo_izq, ojo_der = MainOptAxial(
                    ruta,
                    collector=tiles_finales,
                    guardar_mosaico=guardar_mosaico_por_paciente,
                    y_min_frac=0.2,
                    mostrar_corte_final=mostrar_corte_final,
                    corte_idx_visual=corte_idx_visual,
                    out_dir_mosaico=out_dir_mosaicos,
                    out_dir_paciente=out_dir_paciente
                )
                if ojo_izq is not None:
                    fila["ojo_izq_x"] = float(ojo_izq[0])
                    fila["ojo_izq_y"] = float(ojo_izq[1])
                else:
                    fila["notas"] += "Sin ojo_izq; "
                if ojo_der is not None:
                    fila["ojo_der_x"] = float(ojo_der[0])
                    fila["ojo_der_y"] = float(ojo_der[1])
                else:
                    fila["notas"] += "Sin ojo_der; "
            else:
                ojo_izq = ojo_der = None

            # === 2. Elegir k por porcentaje
            img_itk = sitk.ReadImage(str(ruta))
            img_itk = resample_to_axial_LPS(img_itk)
            Zc = z_center_profile(img_itk)
            zc_min, zc_max = float(Zc.min()), float(Zc.max())

            if target_pct_override is None:
                if (fila["ojo_izq_y"] is not None) and (fila["ojo_der_y"] is not None):
                    rsel = inferir_corte_auto_mm_desde_ojos_pct(
                        volume_path=ruta,
                        corte_idx_ojos=corte_idx_visual,
                        y_ojo1=fila["ojo_izq_y"],
                        y_ojo2=fila["ojo_der_y"],
                        delta_superior_mm=delta_superior_mm,
                        refine_by_center_mm=refine_by_center_mm
                    )
                    if rsel is None:
                        fila["status"] = "WARN"
                        fila["notas"] += "No se pudo inferir Z desde ojos; "
                        if metrics:
                            registros.append(fila)
                        continue

                    fila.update({
                        "k_eye":                 rsel["k_eye"],
                        "k_auto_pct":            rsel["k_auto_pct"],
                        "k_auto":                rsel["k_auto"],
                        "z_eye_mm":              rsel["z_eye_mm"],
                        "z_auto_mm":             rsel["z_auto_mm"],
                        "z_center_min_mm":       rsel["z_center_min_mm"],
                        "z_center_max_mm":       rsel["z_center_max_mm"],
                        "z_auto_mm_snap_center": rsel["z_auto_mm_snap_center"],
                    })
                    fila["pct_auto_target"]      = float(_mm_to_pct(rsel["z_auto_mm"], zc_min, zc_max))
                    fila["pct_auto_snap_center"] = float(_mm_to_pct(rsel["z_auto_mm_snap_center"], zc_min, zc_max))
                    k_final = int(rsel["k_auto"])
                else:
                    fila["status"] = "WARN"
                    fila["notas"] += "Faltan ojos; "
                    k_final = None
            else:
                # Flujo por porcentaje global
                pct = float(_clamp(target_pct_override, 0.0, 1.0))
                k_pct = _pct_to_k(pct, img_itk.GetSize()[2])
                z_target_mm = zc_min + pct * (zc_max - zc_min)
                k_ref = int(np.argmin(np.abs(Zc - z_target_mm))) if refine_by_center_mm else int(k_pct)
                fila.update({
                    "k_auto_pct":            int(k_pct),
                    "k_auto":                int(k_ref),
                    "z_center_min_mm":       zc_min,
                    "z_center_max_mm":       zc_max,
                    "pct_auto_target":       pct,
                    "z_auto_mm":             float(z_target_mm),
                    "z_auto_mm_snap_center": float(Zc[k_ref]),
                    "pct_auto_snap_center":  float(_mm_to_pct(float(Zc[k_ref]), zc_min, zc_max)),
                })
                k_final = int(k_ref)

            # === 3. Guardar corte axial
            if k_final is not None:
                try:
                    nii_path, zc_saved, pct_saved = save_axial_slice_as_nii(
                        volume_path=ruta,
                        k=int(k_final),
                        out_dir=out_dir_nii,
                        paciente_id=paciente_id,
                        zc_range=(zc_min, zc_max),
                        z_label_mm=fila.get("z_auto_mm_snap_center")
                    )
                    fila["nii_auto_path"]     = nii_path
                    fila["z_saved_center_mm"] = float(zc_saved)
                    fila["pct_saved_center"]  = float(pct_saved)
                    print(f"  [OK] Corte guardado en: {nii_path}")
                except Exception as e_save:
                    fila["status"] = "WARN"
                    fila["notas"] += f"No se pudo guardar NIfTI: {e_save}; "
                    print(f"  [WARN] No se pudo guardar el corte: {e_save}")

        except Exception as e:
            fila["status"] = "ERROR"
            fila["notas"]  = str(e)
            print(f"  [ERROR] Paciente {paciente_id} falló: {repr(e)}")

        if metrics:
            registros.append(fila)

    dur = datetime.now() - start
    print(f"\n[BATCH] Finalizado. TOTAL={total} | Duración: {dur}")

    # === 4. Mosaico global (opcional)
    if construir_mosaico_final and usar_collector_en_batch and tiles_finales:
        os.makedirs(out_dir_mosaicos, exist_ok=True)
        Ht, Wt = tiles_finales[0].shape[:2]
        tiles_std = [cv2.resize(im, (Wt, Ht)) for im in tiles_finales]
        cols = min(6, max(1, int(math.sqrt(len(tiles_std)))))
        rows = math.ceil(len(tiles_std) / cols)
        blank = (np.ones_like(tiles_std[0]) * 30).astype(np.uint8)
        tiles_std += [blank] * (rows * cols - len(tiles_std))
        filas_img = [cv2.hconcat(tiles_std[r * cols:(r + 1) * cols]) for r in range(rows)]
        mosaico_final = cv2.vconcat(filas_img)

        out_mosaico = os.path.join(out_dir_mosaicos, "Mosaico_ojos_final_por_paciente.png")
        cv2.imwrite(out_mosaico, mosaico_final)
        print(f"  [OK] Mosaico final por paciente guardado: {out_mosaico}")

        if mostrar_mosaico_final:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(mosaico_final, cv2.COLOR_BGR2RGB))
            plt.title("Mosaico final: corte coronal visual + ojos por paciente")
            plt.axis('off')
            plt.show()

    # === 5. Guardar métricas (opcional)
    if metrics:
        _guardar_metricas(pd.DataFrame(registros, columns=_COLS), metrics_path)