import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation, apply_orientation
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.draw import line
import json
import os
import uuid
import glob
import pandas as pd
from IPython.display import display


# =========================
# === Utilidades imagen  ===
# =========================

def reorient_to_ras(img: nib.Nifti1Image) -> np.ndarray:
    """
    Reorienta un volumen NIfTI al espacio canónico RAS y retorna su array.

    Calcula la transformación entre la orientación original del volumen
    y la orientación RAS (Right-Anterior-Superior), la aplica sobre los
    datos y maneja el caso de imágenes 2D añadiendo una dimensión extra.

    Parámetros
    ----------
    img : nib.Nifti1Image
        Imagen NIfTI cargada con nibabel.

    Retorna
    -------
    np.ndarray
        Array 3D con los datos reorientados a RAS.
    """
    orig_ornt = io_orientation(img.affine)
    ras_ornt  = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(orig_ornt, ras_ornt)
    data = img.get_fdata()
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    return apply_orientation(data, transform)


def ventana(imagen: np.ndarray, nivel: float, ancho: float) -> np.ndarray:
    """
    Aplica una ventana de visualización en escala Hounsfield (HU).

    Recorta los valores de la imagen al rango [nivel - ancho/2, nivel + ancho/2]
    y los normaliza a [0, 255] como uint8. Se usa para realzar el contraste
    de tejidos blandos u óseos según los parámetros elegidos.

    Parámetros
    ----------
    imagen : np.ndarray
        Array con valores en unidades Hounsfield (HU).
    nivel : float
        Centro de la ventana HU (nivel de contraste).
    ancho : float
        Ancho total de la ventana HU.

    Retorna
    -------
    np.ndarray, dtype uint8
        Imagen normalizada en el rango [0, 255].
    """
    min_val = nivel - (ancho / 2)
    max_val = nivel + (ancho / 2)
    ventana_vals = np.clip(imagen, min_val, max_val)
    ventana_vals = np.float64((ventana_vals - min_val) / (max_val - min_val) * 255)
    return ventana_vals.astype(np.uint8)


# =========================
# === Líneas tangenciales==
# =========================

def punto_mas_arriba_e_izquierda(mask_hueso: np.ndarray) -> tuple:
    """
    Calcula los dos puntos que definen la línea tangencial izquierda del cráneo.

    Recorre la máscara ósea de arriba hacia abajo para encontrar el punto
    superior-izquierdo del cráneo (p1), y luego busca el segundo punto (p2)
    minimizando los píxeles de hueso que la línea p1→p2 atraviesa, lo que
    garantiza que la recta sea tangente al contorno craneal.

    Parámetros
    ----------
    mask_hueso : np.ndarray
        Máscara binaria 2D de la región ósea craneal.

    Retorna
    -------
    tuple of (np.ndarray, np.ndarray, list)
        - Punto superior (x, y) en coordenadas de imagen.
        - Punto inferior (x, y) en coordenadas de imagen.
        - Lista de candidatos evaluados como pares (x, y).
    """
    punto = np.array([-1, 1000])
    romper = False

    for i in range(0, mask_hueso.shape[0], 5):
        for j in range(mask_hueso.shape[1]):
            if mask_hueso[i, j] == 1:
                if i > punto[0] and j < punto[1]:
                    punto = np.array([i, j])
                    break
                elif i > punto[0] and j > punto[1]:
                    romper = True
        if romper:
            break

    num = 100
    punto2 = np.array([0, 0])
    candidatos = []

    for i in range(mask_hueso.shape[0] - 1, punto[0] + 20, -1):
        for j in range(mask_hueso.shape[1]):
            if mask_hueso[i, j] == 1:
                puntor = np.array([i, j])
                candidatos.append(puntor.copy())

                rr, cc = line(punto[0], punto[1], puntor[0], puntor[1])
                valores = mask_hueso[rr, cc]
                valores = valores[mask_hueso[rr, cc] == 1]

                if len(valores) < num:
                    num = len(valores)
                    punto2 = puntor
                else:
                    break

    return (
        np.array([punto[1],  punto[0]]),
        np.array([punto2[1], punto2[0]]),
        [(p[1], p[0]) for p in candidatos]
    )


def punto_mas_arriba_y_derecha(mask_hueso: np.ndarray) -> tuple:
    """
    Calcula los dos puntos que definen la línea tangencial derecha del cráneo.

    Versión simétrica de punto_mas_arriba_e_izquierda, recorre la máscara
    de arriba hacia abajo y de derecha a izquierda para encontrar el punto
    superior-derecho del cráneo y su segundo punto tangencial.

    Parámetros
    ----------
    mask_hueso : np.ndarray
        Máscara binaria 2D de la región ósea craneal.

    Retorna
    -------
    tuple of (np.ndarray, np.ndarray, list)
        - Punto superior (x, y) en coordenadas de imagen.
        - Punto inferior (x, y) en coordenadas de imagen.
        - Lista de candidatos evaluados como pares (x, y).
    """
    punto = np.array([-1, -1])
    romper = False

    for i in range(0, mask_hueso.shape[0], 5):
        for j in range(mask_hueso.shape[1] - 1, 0, -1):
            if mask_hueso[i, j] == 1:
                if i > punto[0] and j > punto[1]:
                    punto = np.array([i, j])
                    break
                elif i > punto[0] and j < punto[1]:
                    romper = True
        if romper:
            break

    num = 100
    punto2 = np.array([0, 0])
    candidatos = []

    for i in range(mask_hueso.shape[0] - 1, punto[0] + 20, -1):
        for j in range(mask_hueso.shape[1] - 1, 0, -1):
            if mask_hueso[i, j] == 1:
                puntor = np.array([i, j])
                candidatos.append(puntor.copy())

                rr, cc = line(punto[0], punto[1], puntor[0], puntor[1])
                valores = mask_hueso[rr, cc]
                valores = valores[mask_hueso[rr, cc] == 1]

                if len(valores) < num:
                    num = len(valores)
                    punto2 = puntor
                else:
                    break

    return (
        np.array([punto[1],  punto[0]]),
        np.array([punto2[1], punto2[0]]),
        [(p[1], p[0]) for p in candidatos]
    )


# =========================
# === Grosor muscular    ===
# =========================

def calcular_grosor_musculo(p1, p2, mask_musculo: np.ndarray,
                             pixel_spacing: tuple,
                             N: int = 100,
                             L: int = 100,
                             extension_factor: float = 1.2) -> tuple:
    """
    Calcula el grosor máximo del músculo temporal a lo largo de una tangente.

    Genera N puntos a lo largo de la línea tangencial extendida p1→p2,
    traza perpendiculares de longitud 2L en cada punto, mide el segmento
    continuo de músculo que cada perpendicular atraviesa y retorna la
    medición máxima en mm junto con las coordenadas del segmento óptimo.

    Parámetros
    ----------
    p1 : array-like, shape (2,)
        Punto superior de la línea tangencial (x, y).
    p2 : array-like, shape (2,)
        Punto inferior de la línea tangencial (x, y).
    mask_musculo : np.ndarray
        Máscara binaria 2D de la región muscular temporal.
    pixel_spacing : tuple of (float, float)
        Tamaño real de cada píxel en mm (dx, dy) según el header NIfTI.
    N : int, opcional
        Número de puntos de muestreo a lo largo de la tangente. Por defecto 100.
    L : int, opcional
        Semi-longitud en píxeles de cada perpendicular. Por defecto 100.
    extension_factor : float, opcional
        Factor de extensión de la tangente más allá de p2. Por defecto 1.2.

    Retorna
    -------
    tuple of (float, tuple or None)
        - Grosor máximo del músculo en mm.
        - Coordenadas del segmento óptimo como ((x0, y0), (x1, y1)),
          o None si no se encontró ningún segmento válido.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    v_tan  = p2 - p1
    v_tan  = v_tan / np.linalg.norm(v_tan)
    v_perp = np.array([-v_tan[1], v_tan[0]])

    extension_length = np.linalg.norm(p2 - p1) * extension_factor
    p3_extendido = p1 + v_tan * extension_length
    puntos = [p1 + t * (p3_extendido - p1) / N for t in range(N)]

    mejor_longitud_mm = 0.0
    mejor_linea = None

    dx_mm, dy_mm = pixel_spacing

    for punto in puntos:
        punto  = np.round(punto).astype(int)
        inicio = np.round(punto - L * v_perp).astype(int)
        fin    = np.round(punto + L * v_perp).astype(int)

        rr, cc = line(inicio[1], inicio[0], fin[1], fin[0])
        rr = np.clip(rr, 0, mask_musculo.shape[0] - 1)
        cc = np.clip(cc, 0, mask_musculo.shape[1] - 1)

        valores  = mask_musculo[rr, cc].astype(bool)
        indices  = np.where(valores)[0]

        if len(indices) >= 2:
            cortes = np.where(np.diff(indices) > 1)[0]

            if len(cortes) > 0:
                idx_inicio = indices[0]
                idx_fin    = indices[cortes[0]]
            else:
                idx_inicio = indices[0]
                idx_fin    = indices[-1]

            y0, x0 = rr[idx_inicio], cc[idx_inicio]
            y1, x1 = rr[idx_fin],    cc[idx_fin]

            dist_mm = np.sqrt(((x1 - x0) * dx_mm) ** 2 + ((y1 - y0) * dy_mm) ** 2)

            if dist_mm > mejor_longitud_mm:
                mejor_longitud_mm = dist_mm
                mejor_linea = ((x0, y0), (x1, y1))

    return mejor_longitud_mm, mejor_linea


# =========================
# === Exportar markups   ===
# =========================

def pixel_a_mundo(x: float, y: float, z: float, affine: np.ndarray) -> list:
    """
    Convierte coordenadas de píxel (i, j, k) a coordenadas físicas RAS (mm).

    Parámetros
    ----------
    x : float
        Índice de columna (eje X de imagen).
    y : float
        Índice de fila (eje Y de imagen).
    z : float
        Índice de corte axial (eje Z de imagen).
    affine : np.ndarray, shape (4, 4)
        Matriz affine del volumen NIfTI.

    Retorna
    -------
    list of float
        Coordenadas físicas [X, Y, Z] en mm en el espacio RAS.
    """
    ijk = np.array([x, y, z, 1.0])
    xyz = affine @ ijk
    return xyz[:3].tolist()


def crear_markup_line(nombre: str, p0: tuple, p1: tuple,
                       affine: np.ndarray, slice_z: int) -> dict:
    """
    Construye un markup de tipo línea en formato compatible con 3D Slicer.

    Convierte dos puntos en coordenadas de píxel a coordenadas físicas RAS
    y los empaqueta en la estructura JSON que utiliza el módulo Markups
    de 3D Slicer.

    Parámetros
    ----------
    nombre : str
        Nombre identificador del markup (aparece en la lista de markups de Slicer).
    p0 : tuple of (float, float)
        Primer punto en coordenadas de píxel (x, y).
    p1 : tuple of (float, float)
        Segundo punto en coordenadas de píxel (x, y).
    affine : np.ndarray, shape (4, 4)
        Matriz affine del volumen NIfTI.
    slice_z : int
        Índice del corte axial del que provienen los puntos.

    Retorna
    -------
    dict
        Diccionario con la estructura de un markup de línea para 3D Slicer.
    """
    return {
        "type": "Line",
        "coordinateSystem": "RAS",
        "locked": False,
        "labelFormat": "%N",
        "name": nombre,
        "controlPoints": [
            {
                "id": str(uuid.uuid4()),
                "position": pixel_a_mundo(p0[0], p0[1], slice_z, affine),
                "orientation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "visibility": True
            },
            {
                "id": str(uuid.uuid4()),
                "position": pixel_a_mundo(p1[0], p1[1], slice_z, affine),
                "orientation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "visibility": True
            }
        ]
    }


def exportar_TMT_a_markups(ruta_nifti: str,
                            out_json: str,
                            p1, p2,
                            p1D, p2D,
                            mejor_lineaI,
                            mejor_lineaD) -> None:
    """
    Exporta las mediciones TMT como archivo de markups JSON para 3D Slicer.

    Crea un archivo .mrk.json con las líneas perpendiculares izquierda y
    derecha en coordenadas físicas RAS, listo para importar en 3D Slicer
    con el módulo Markups.

    Parámetros
    ----------
    ruta_nifti : str
        Ruta al archivo NIfTI del corte axial, usado para obtener la
        matriz affine.
    out_json : str
        Ruta de salida para el archivo .mrk.json.
    p1 : array-like
        Punto superior de la tangente izquierda (x, y).
    p2 : array-like
        Punto inferior de la tangente izquierda (x, y).
    p1D : array-like
        Punto superior de la tangente derecha (x, y).
    p2D : array-like
        Punto inferior de la tangente derecha (x, y).
    mejor_lineaI : tuple or None
        Segmento óptimo izquierdo como ((x0, y0), (x1, y1)).
    mejor_lineaD : tuple or None
        Segmento óptimo derecho como ((x0, y0), (x1, y1)).

    Retorna
    -------
    None
    """
    nii    = nib.load(ruta_nifti)
    affine = nii.affine
    slice_z = 0
    markups = []

    if mejor_lineaI is not None:
        (x0, y0), (x1, y1) = mejor_lineaI
        markups.append(crear_markup_line(
            "Perpendicular_Derecha", (x0, y0), (x1, y1), affine, slice_z
        ))

    if mejor_lineaD is not None:
        (x0, y0), (x1, y1) = mejor_lineaD
        markups.append(crear_markup_line(
            "Perpendicular_Izquierda", (x0, y0), (x1, y1), affine, slice_z
        ))

    estructura = {
        "@schema": (
            "https://raw.githubusercontent.com/slicer/slicer/master/"
            "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json"
        ),
        "markups": markups
    }

    with open(out_json, "w") as f:
        json.dump(estructura, f, indent=2)

    print(f"  [OK] Markups exportados en: {out_json}")


# =========================
# === Procesamiento TMT  ===
# =========================

def Processing(ruta: str) -> tuple:
    """
    Ejecuta el pipeline completo de segmentación y medición TMT en un corte axial.

    1) Carga el NIfTI, extrae el spacing real y reorienta a RAS.
    2) Segmenta hueso (HU >= 500) y músculo (20 <= HU <= 200) por umbral.
    3) Filtra los componentes musculares para conservar solo los dos músculos
       temporales, ignorando el cerebro (componente de mayor área central).
    4) Aplica morfología de cierre al hueso craneal y selecciona el componente
       de mayor área.
    5) Calcula las líneas tangenciales izquierda y derecha del cráneo.
    6) Mide el grosor muscular máximo perpendicular a cada tangente en mm reales.
    7) Muestra una figura con la imagen original, las máscaras y las mediciones.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo NIfTI (.nii o .nii.gz) del corte axial óptimo.

    Retorna
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              float, tuple or None, float, tuple or None)
        - data           : array 2D del corte axial en HU.
        - mask_hueso     : máscara binaria 2D del cráneo.
        - mask_musculo   : máscara binaria 2D de los músculos temporales.
        - p1             : punto superior tangente izquierda (x, y).
        - p2             : punto inferior tangente izquierda (x, y).
        - p1D            : punto superior tangente derecha (x, y).
        - p2D            : punto inferior tangente derecha (x, y).
        - mejor_longitudI: grosor músculo izquierdo en mm.
        - mejor_lineaI   : segmento óptimo izquierdo ((x0,y0),(x1,y1)) o None.
        - mejor_longitudD: grosor músculo derecho en mm.
        - mejor_lineaD   : segmento óptimo derecho ((x0,y0),(x1,y1)) o None.

    Lanza
    -----
    Retorna una tupla con valores None / 0 si no se pueden calcular los
    puntos tangenciales.
    """
    # === 1. Carga y reorientación
    nii     = nib.load(ruta)
    spacing = nii.header.get_zooms()[:2]  # (dx, dy) en mm
    data    = reorient_to_ras(nii)
    data    = np.rot90(data, k=1)
    data    = np.fliplr(data)
    data    = data[:, :, 0]

    # === 2. Segmentación por umbral HU
    mask_hueso   = (data >= 500).astype(np.uint8)
    mask_musculo = ((data >= 20) & (data <= 200)).astype(np.uint8)

    # === 3. Filtrado de músculos temporales (ignorar cerebro)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_musculo, connectivity=4
    )
    H = mask_musculo.shape[0]
    y_min, y_max = 0.30 * H, 0.70 * H

    candidatos = sorted(
        [(lab, stats[lab, 4]) for lab in range(1, num_labels)
         if y_min <= centroids[lab][1] <= y_max],
        key=lambda x: x[1], reverse=True
    )

    if len(candidatos) >= 3:
        labels_sel = [candidatos[1][0], candidatos[2][0]]
    elif len(candidatos) == 2:
        labels_sel = [candidatos[0][0], candidatos[1][0]]
    else:
        print("  [WARN] Fallback: selección global por área en máscara muscular.")
        areas = stats[1:, 4]
        idx = np.argsort(areas)
        labels_sel = list(idx[-3:-1] + 1)

    mask_musculo_filtrada = np.zeros_like(mask_musculo, dtype=np.uint8)
    for lab in labels_sel:
        mask_musculo_filtrada[labels == lab] = 1
    mask_musculo = mask_musculo_filtrada

    # === 4. Selección del componente craneal principal
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_hueso, connectivity=4
    )
    indice_max = np.argmax(stats[1:, 4]) + 1
    mask_craneo = np.zeros(mask_hueso.shape)
    mask_craneo[labels == indice_max] = 1
    mask_hueso = mask_craneo

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask_hueso = cv2.morphologyEx(mask_hueso, cv2.MORPH_CLOSE, kernel)

    # === 5. Líneas tangenciales
    p1,  p2,  _ = punto_mas_arriba_e_izquierda(mask_hueso)
    p1D, p2D, _ = punto_mas_arriba_y_derecha(mask_hueso)

    if p1 is None or p2 is None or p1D is None or p2D is None:
        print("  [ERROR] No se pudieron calcular los puntos tangenciales.")
        return None, None, None, None, None, None, None, 0, None, 0, None

    # === 6. Medición del grosor muscular
    mejor_longitudI, mejor_lineaI = calcular_grosor_musculo(
        p1, p2, mask_musculo, pixel_spacing=spacing
    )
    mejor_longitudD, mejor_lineaD = calcular_grosor_musculo(
        p1D, p2D, mask_musculo, pixel_spacing=spacing
    )

    # === 7. Visualización
    data_vis = ventana(data, nivel=50, ancho=100)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))

    axs[0].imshow(data_vis, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(mask_hueso, cmap='gray')
    axs[1].set_title('Cráneo')
    axs[1].axis('off')

    axs[2].imshow(mask_musculo, cmap='gray')
    axs[2].set_title('Músculo')
    axs[2].axis('off')

    axs[3].imshow(data_vis, cmap='gray')
    axs[3].set_title('Automatic TMT')
    axs[3].plot([p1[0],  p2[0]],  [p1[1],  p2[1]],  'g',    linewidth=1)
    axs[3].plot([p1D[0], p2D[0]], [p1D[1], p2D[1]], 'pink', linewidth=1)

    if mejor_lineaI is not None:
        (i0, i1) = mejor_lineaI
        axs[3].plot([i0[0], i1[0]], [i0[1], i1[1]], 'r', linewidth=2)
        axs[3].text(i0[0], i0[1] - 10, f"{mejor_longitudI:.1f} mm", color='r')

    if mejor_lineaD is not None:
        (i0, i1) = mejor_lineaD
        axs[3].plot([i0[0], i1[0]], [i0[1], i1[1]], 'b', linewidth=2)
        axs[3].text(i0[0], i0[1] - 10, f"{mejor_longitudD:.1f} mm", color='b')

    axs[3].axis('off')
    plt.tight_layout()
    plt.show()

    return (
        data,
        mask_hueso,
        mask_musculo,
        p1, p2,
        p1D, p2D,
        mejor_longitudI, mejor_lineaI,
        mejor_longitudD, mejor_lineaD
    )


# =========================
# === MAIN Individual   ===
# =========================

def MainTMT_Measure(ruta_archivo: str, path_output_csv: str) -> None:
    """
    Mide el grosor del músculo temporal en un único corte axial y guarda los resultados.

    Ejecuta Processing sobre el archivo indicado, construye una fila de
    resultados con el nombre del paciente, las coordenadas de los segmentos
    y los grosores medidos, guarda un CSV individual y muestra el DataFrame.

    Parámetros
    ----------
    ruta_archivo : str
        Ruta al archivo NIfTI (.nii o .nii.gz) del corte axial óptimo.
    path_output_csv : str
        Carpeta donde se guardará el archivo CSV de resultados.

    Retorna
    -------
    None
    """
    nombre_archivo = os.path.basename(ruta_archivo).split(".")[0].removeprefix(
        "Corte_axial_optimo_paciente_"
    )
    print(f"\n[SINGLE] Procesando paciente: {nombre_archivo}")

    try:
        _, _, _, p1, p2, p1D, p2D, mejor_longitudI, _, mejor_longitudD, _ = Processing(ruta_archivo)

        grosor_promedio = (mejor_longitudI + mejor_longitudD) / 2

        fila = {
            "scan":            nombre_archivo,
            "coordinates_R":   f"{p1} a {p2}",
            "coordinates_L":   f"{p1D} a {p2D}",
            "Thickness_R":     round(mejor_longitudI,   2),
            "Thickness_L":     round(mejor_longitudD,   2),
            "Mean Thickness":  round(grosor_promedio,   2)
        }

    except Exception as e:
        print(f"  [ERROR] Paciente {nombre_archivo} falló: {repr(e)}")
        fila = {
            "scan":           nombre_archivo,
            "coordinates_R":  "error",
            "coordinates_L":  "error",
            "Thickness_R":    "error",
            "Thickness_L":    "error",
            "Mean Thickness": "error"
        }

    df_resultados = pd.DataFrame([fila])
    out_csv = os.path.join(
        path_output_csv, f"resultados_temporales_{nombre_archivo}.csv"
    )
    os.makedirs(path_output_csv, exist_ok=True)
    df_resultados.to_csv(out_csv, index=False)

    print(f"  [OK] CSV guardado en: {out_csv}")
    display(df_resultados)


# =========================
# === MAIN Batch        ===
# =========================

def MainTMT_Measure_batch(input_paths_or_dir,
                           path_output_csv: str,
                           recursive: bool = False,
                           guardar_csv_individual: bool = False,
                           nombre_csv_final: str = "resultados_TMT_batch.csv") -> pd.DataFrame:
    """
    Ejecuta la medición TMT en batch para múltiples cortes axiales.

    Recorre todos los archivos NIfTI encontrados en input_paths_or_dir,
    llama a Processing en cada uno, acumula los resultados y guarda un
    CSV consolidado al finalizar. Opcionalmente guarda también un CSV
    individual por paciente.

    Parámetros
    ----------
    input_paths_or_dir : str o list of str
        Carpeta con cortes axiales (.nii / .nii.gz) o lista de rutas.
    path_output_csv : str
        Carpeta donde se guardarán los archivos CSV de salida.
    recursive : bool, opcional
        Si es True, busca NIfTI recursivamente en subdirectorios.
        Por defecto False.
    guardar_csv_individual : bool, opcional
        Si es True, guarda un CSV por paciente además del consolidado.
        Por defecto False.
    nombre_csv_final : str, opcional
        Nombre del archivo CSV consolidado. Por defecto
        "resultados_TMT_batch.csv".

    Retorna
    -------
    pd.DataFrame
        DataFrame con una fila por paciente y columnas: scan,
        coordinates_R, coordinates_L, Thickness_R, Thickness_L,
        Mean Thickness.
    """
    os.makedirs(path_output_csv, exist_ok=True)

    # --- Recolectar archivos NIfTI ---
    def _gather(p):
        files = []
        targets = p if isinstance(p, (list, tuple)) else [p]
        for t in targets:
            if os.path.isdir(t):
                pat1 = os.path.join(t, "**", "*.nii")    if recursive else os.path.join(t, "*.nii")
                pat2 = os.path.join(t, "**", "*.nii.gz") if recursive else os.path.join(t, "*.nii.gz")
                files.extend(glob.glob(pat1, recursive=recursive))
                files.extend(glob.glob(pat2, recursive=recursive))
            elif os.path.isfile(t):
                files.append(t)
        return sorted(files)

    archivos = _gather(input_paths_or_dir)

    _COLS = ["scan", "coordinates_R", "coordinates_L",
             "Thickness_R", "Thickness_L", "Mean Thickness"]

    if not archivos:
        print("[BATCH] No se encontraron archivos NIfTI para procesar.")
        return pd.DataFrame(columns=_COLS)

    total = len(archivos)
    print(f"[BATCH] Archivos detectados: {total}")
    print(f"[BATCH] Entrada: {input_paths_or_dir}")
    print(f"[BATCH] Salida:  {path_output_csv}")

    filas  = []
    ok     = 0
    fail   = 0
    from datetime import datetime
    start  = datetime.now()

    for i, ruta in enumerate(archivos, 1):
        nombre_archivo = os.path.basename(ruta).split(".")[0].removeprefix(
            "Corte_axial_optimo_paciente_"
        )
        print(f"\n[{i}/{total}] Procesando paciente: {nombre_archivo}")

        try:
            _, _, _, p1, p2, p1D, p2D, mejor_longitudI, _, mejor_longitudD, _ = Processing(ruta)

            grosor_promedio = (mejor_longitudI + mejor_longitudD) / 2

            fila = {
                "scan":           nombre_archivo,
                "coordinates_R":  f"{p1} a {p2}",
                "coordinates_L":  f"{p1D} a {p2D}",
                "Thickness_R":    round(mejor_longitudI,  2),
                "Thickness_L":    round(mejor_longitudD,  2),
                "Mean Thickness": round(grosor_promedio,  2)
            }
            ok += 1
            print(f"  [OK] Paciente {nombre_archivo} medido. "
                  f"R={fila['Thickness_R']} mm | L={fila['Thickness_L']} mm | "
                  f"Media={fila['Mean Thickness']} mm")

        except Exception as e:
            print(f"  [ERROR] Paciente {nombre_archivo} falló: {repr(e)}")
            fila = {
                "scan":           nombre_archivo,
                "coordinates_R":  "error",
                "coordinates_L":  "error",
                "Thickness_R":    "error",
                "Thickness_L":    "error",
                "Mean Thickness": "error"
            }
            fail += 1

        filas.append(fila)

        if guardar_csv_individual:
            df_tmp  = pd.DataFrame([fila], columns=_COLS)
            out_ind = os.path.join(
                path_output_csv, f"resultados_temporales_{fila['scan']}.csv"
            )
            df_tmp.to_csv(out_ind, index=False)

    dur = datetime.now() - start
    print(f"\n[BATCH] Finalizado. OK={ok} | FAIL={fail} | TOTAL={total} | Duración: {dur}")

    df_final = pd.DataFrame(filas, columns=_COLS)

    out_csv = os.path.join(path_output_csv, nombre_csv_final)
    df_final.to_csv(out_csv, index=False)
    print(f"[BATCH] CSV consolidado guardado en: {out_csv}")

    display(df_final)
    return df_final