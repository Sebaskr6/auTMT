import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from sklearn.decomposition import PCA
import os


def MainReorientation(ruta, output, show=True):
    """
    Reorienta el volumen CT de un paciente y lo guarda como NIfTI (.nii.gz).

    Carga una serie DICOM desde 'ruta', calcula los ángulos de rotación
    mediante PCA en los tres planos anatómicos (YZ, XZ, XY), aplica la
    rotación combinada en el header del volumen y guarda el resultado.

    Parámetros
    ----------
    ruta : str
        Ruta a la carpeta que contiene la serie DICOM del paciente.
    output : str
        Ruta a la carpeta de salida donde se guardará el archivo .nii.gz.
    show : bool, opcional
        Si es True, muestra las vistas multiplanares antes y después de
        la reorientación. Por defecto True.

    Retorna
    -------
    sitk.Image
        Volumen reorientado con orientación LPS en el header.
    """
    ## === 0. Cargar y mostrar volumen original
    volumen_original = cargar_volumen_estandar(ruta)
    nombre_archivo = os.path.basename(os.path.normpath(ruta)).split(".")[0]

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        mostrar_vistas_multiplanares(volumen_original, axs, "Volumen Original")
        plt.tight_layout()
        plt.show()

    ## === 1. Reorientación craneal
    ang_x = proyectar_plano(volumen_original, "YZ")  # PCA
    ang_y = proyectar_plano(volumen_original, "XZ")
    ang_z = proyectar_plano(volumen_original, "XY")

    if ang_y > 90:
        ang_y = 90 - ang_y  # Corrección ángulo
    else:
        ang_y = -ang_y

    rot_x = obtener_rotacion_eje_x(-ang_x)
    rot_y = np.eye(3)
    rot_z = obtener_rotacion_eje_z(90 - ang_z)

    # IMPORTANTE: usar transpuesta (convención)
    R_total = (rot_z @ rot_y @ rot_x).T
    volumen_reorientado = aplicar_rotacion_en_header(volumen_original, R_total)

    # Preview solo para visualizar (hardened)
    if show:
        volumen_preview = harden_to_identity(volumen_reorientado, default_value=-1024)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        mostrar_vistas_multiplanares_2(volumen_preview, axs, "Volumen Reorientado")
        plt.tight_layout()
        plt.show()

    # === Guardar el volumen reorientado como NIfTI (.nii.gz)
    volumen_reorientado = sitk.DICOMOrient(volumen_reorientado, "LPS")

    guardar_volumen_como_nii(volumen_reorientado, output, nombre_archivo)

    return volumen_reorientado


# =========================
# === BATCH MAIN
# =========================
def find_dicom_series_dir(patient_dir, max_depth=2):
    """
    Busca la subcarpeta más superficial que contenga una serie DICOM legible.

    Recorre en anchura (BFS) desde 'patient_dir' hasta 'max_depth' niveles
    de profundidad y devuelve la primera carpeta en la que SimpleITK detecte
    una serie DICOM válida.

    Parámetros
    ----------
    patient_dir : str
        Ruta raíz desde donde iniciar la búsqueda.
    max_depth : int, opcional
        Número máximo de niveles de subdirectorios a explorar. Por defecto 2.

    Retorna
    -------
    str or None
        Ruta absoluta de la carpeta con la serie DICOM, o None si no se
        encuentra ninguna dentro del rango de profundidad indicado.
    """
    patient_dir = os.path.abspath(patient_dir)

    def has_series(d):
        try:
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(d)
            return ids is not None and len(ids) > 0
        except Exception:
            return False

    if has_series(patient_dir):
        return patient_dir

    # BFS hasta profundidad max_depth
    current = [(patient_dir, 0)]
    visited = set()

    while current:
        d, depth = current.pop(0)
        if d in visited:
            continue
        visited.add(d)

        if depth >= max_depth:
            continue

        try:
            subdirs = [os.path.join(d, sd) for sd in os.listdir(d) if os.path.isdir(os.path.join(d, sd))]
        except Exception:
            subdirs = []

        for sd in subdirs:
            if has_series(sd):
                return sd
            current.append((sd, depth + 1))

    return None


def MainReorientationBatch(root_patients_dir, output_dir, show=False, max_depth=2):
    """
    Ejecuta la reorientación de forma secuencial para un conjunto de pacientes.

    Itera sobre las subcarpetas de 'root_patients_dir', localiza la serie
    DICOM de cada paciente con 'find_dicom_series_dir' y llama a
    'MainReorientation'. Registra en consola el progreso y un resumen final
    con el número de pacientes procesados correctamente y los que fallaron.

    Estructura esperada de directorios
    -----------------------------------
    root_patients_dir/
        0/  (o 0/todoslosdcm/)
        1/
        2/
        ...

    Parámetros
    ----------
    root_patients_dir : str
        Ruta a la carpeta raíz que contiene una subcarpeta por paciente.
    output_dir : str
        Ruta a la carpeta de salida donde se guardarán los archivos .nii.gz.
    show : bool, opcional
        Si es True, muestra las figuras de visualización por cada paciente.
        Se recomienda False para procesamiento batch. Por defecto False.
    max_depth : int, opcional
        Profundidad máxima de búsqueda de la serie DICOM dentro de cada
        carpeta de paciente. Por defecto 2.

    Retorna
    -------
    None
    """
    root_patients_dir = os.path.abspath(root_patients_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patient_folders = [
        os.path.join(root_patients_dir, d)
        for d in os.listdir(root_patients_dir)
        if os.path.isdir(os.path.join(root_patients_dir, d))
    ]
    patient_folders.sort(key=lambda p: os.path.basename(p))

    total = len(patient_folders)
    print(f"[BATCH] Pacientes detectados: {total}")
    print(f"[BATCH] Entrada: {root_patients_dir}")
    print(f"[BATCH] Salida:  {output_dir}")

    ok = 0
    fail = 0

    for i, pdir in enumerate(patient_folders, start=1):
        pid = os.path.basename(os.path.normpath(pdir))
        print(f"\n[{i}/{total}] Procesando paciente: {pid}")

        dicom_dir = find_dicom_series_dir(pdir, max_depth=max_depth)
        if dicom_dir is None:
            print(f"  [ERROR] No se encontró serie DICOM en: {pdir}")
            fail += 1
            continue

        try:
            _ = MainReorientation(dicom_dir, output_dir, show=show)
            ok += 1
            print(f"  [OK] Paciente {pid} reorientado y guardado.")
        except Exception as e:
            fail += 1
            print(f"  [ERROR] Paciente {pid} falló: {repr(e)}")

    print(f"\n[BATCH] Finalizado. OK={ok} | FAIL={fail} | TOTAL={total}")


# =========================
# === Rotación física (preview)
# =========================
def harden_to_identity(img, default_value=-1024.0):
    """
    Remuestrea el volumen a una grilla con dirección identidad (ejes alineados).

    Calcula el bounding box físico del volumen rotado, construye una imagen
    de referencia con spacing original y dirección identidad, y aplica un
    resample lineal. Se usa exclusivamente para generar previsualizaciones
    correctas tras modificar el header; no altera el volumen de trabajo.

    Parámetros
    ----------
    img : sitk.Image
        Volumen de entrada, posiblemente con dirección no identidad.
    default_value : float, opcional
        Valor de relleno para vóxeles fuera del FOV original (HU).
        Por defecto -1024.0 (aire en CT).

    Retorna
    -------
    sitk.Image
        Volumen remuestrado con dirección identidad y mismo spacing que
        la imagen de entrada.
    """
    spacing = np.array(img.GetSpacing(), dtype=float)
    size = np.array(img.GetSize(), dtype=int)  # (X,Y,Z)

    corners = np.array([
        [0, 0, 0],
        [size[0]-1, 0, 0],
        [0, size[1]-1, 0],
        [0, 0, size[2]-1],
        [size[0]-1, size[1]-1, 0],
        [size[0]-1, 0, size[2]-1],
        [0, size[1]-1, size[2]-1],
        [size[0]-1, size[1]-1, size[2]-1],
    ], dtype=np.int64)

    phys = np.array([
        img.TransformIndexToPhysicalPoint([int(c[0]), int(c[1]), int(c[2])])
        for c in corners
    ], dtype=float)

    minp = phys.min(axis=0)
    maxp = phys.max(axis=0)

    out_size = np.ceil((maxp - minp) / spacing).astype(int) + 1
    out_size = [int(out_size[0]), int(out_size[1]), int(out_size[2])]

    ref = sitk.Image(out_size, img.GetPixelID())
    ref.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    ref.SetOrigin((float(minp[0]), float(minp[1]), float(minp[2])))
    ref.SetDirection((1, 0, 0,
                      0, 1, 0,
                      0, 0, 1))

    T = sitk.AffineTransform(3)  # identidad
    out = sitk.Resample(img, ref, T, sitk.sitkLinear, float(default_value))
    return out


# =========================
# === Rotación en header
# =========================
def aplicar_rotacion_en_header(img, R):
    """
    Aplica una rotación rígida al volumen modificando únicamente su header.

    Actualiza la matriz de dirección y el origen del volumen de forma que
    la rotación se realice alrededor del centro físico del volumen, sin
    interpolar ni mover los datos de los vóxeles.

    Parámetros
    ----------
    img : sitk.Image
        Volumen de entrada sobre el que se aplicará la rotación.
    R : np.ndarray, shape (3, 3)
        Matriz de rotación 3×3 ortogonal.

    Retorna
    -------
    sitk.Image
        Copia del volumen con dirección y origen actualizados.
    """
    D = np.array(img.GetDirection()).reshape(3, 3)
    O = np.array(img.GetOrigin(), dtype=float)

    size = np.array(img.GetSize(), dtype=float)  # (X,Y,Z)
    center_idx = (size - 1) / 2.0
    C = np.array(img.TransformContinuousIndexToPhysicalPoint(center_idx.tolist()), dtype=float)

    D2 = R @ D
    O2 = C + (R @ (O - C))

    out = sitk.Image(img)
    out.SetDirection(D2.flatten().tolist())
    out.SetOrigin(tuple(O2))
    return out


# =========================
# === Cargar volumen DICOM
# =========================
def cargar_volumen_estandar(dicom_folder):
    """
    Carga una serie DICOM y la orienta al espacio estándar RAI.

    Lee la primera serie DICOM encontrada en 'dicom_folder' usando
    SimpleITK y aplica orientación RAI (Right-Anterior-Inferior) como
    punto de partida consistente para todos los volúmenes.

    Parámetros
    ----------
    dicom_folder : str
        Ruta a la carpeta que contiene los archivos DICOM de la serie.

    Retorna
    -------
    sitk.Image
        Volumen cargado y reorientado a RAI.

    Lanza
    -----
    FileNotFoundError
        Si no se encuentra ninguna serie DICOM válida en la carpeta.
    """
    dicom_folder = os.path.abspath(dicom_folder)
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        raise FileNotFoundError(f"No se encontró serie DICOM en: {dicom_folder}")

    series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)
    image = reader.Execute()
    return sitk.DICOMOrient(image, "RAI")


# =========================
# === Obtener matrices de orientación
# =========================
def obtener_matriz_orientacion(volumen):
    """
    Extrae la matriz de dirección (cosenos directores) del volumen.

    Parámetros
    ----------
    volumen : sitk.Image
        Volumen del cual se extrae la información de orientación.

    Retorna
    -------
    np.ndarray, shape (3, 3)
        Matriz de dirección del volumen, donde cada columna corresponde
        al vector unitario de un eje (X, Y, Z) en coordenadas físicas.
    """
    matriz = np.array(volumen.GetDirection()).reshape(3, 3)
    return matriz


# =========================
# === Visualización
# =========================
def mostrar_vistas_multiplanares(vol_sitk, axs, titulo="Volumen"):
    """
    Dibuja los cortes central axial, coronal y sagital del volumen.

    Toma el slice central en cada dimensión y lo muestra en escala de
    grises sobre los tres ejes de la figura recibida.

    Parámetros
    ----------
    vol_sitk : sitk.Image
        Volumen a visualizar.
    axs : array-like of matplotlib.axes.Axes, longitud 3
        Ejes de matplotlib donde se dibujarán los tres cortes.
    titulo : str, opcional
        Prefijo del título de cada subplot. Por defecto "Volumen".

    Retorna
    -------
    None
    """
    array = sitk.GetArrayFromImage(vol_sitk)
    z, y, x = array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2

    axs[0].imshow(array[z], cmap='gray')
    axs[0].set_title(f"{titulo} - Axial")
    axs[0].axis('off')

    axs[1].imshow(array[:, y, :], cmap='gray')
    axs[1].set_title(f"{titulo} - Coronal")
    axs[1].axis('off')

    axs[2].imshow(array[:, :, x], cmap='gray')
    axs[2].set_title(f"{titulo} - Sagital")
    axs[2].axis('off')


def mostrar_vistas_multiplanares_2(vol_sitk, axs, titulo="Volumen", rot180=True):
    """
    Dibuja los cortes central axial, coronal y sagital con rotación opcional.

    Versión alternativa de 'mostrar_vistas_multiplanares' que permite
    rotar 180° cada corte antes de mostrarlo, útil para corregir la
    orientación visual tras remuestrear a dirección identidad.

    Parámetros
    ----------
    vol_sitk : sitk.Image
        Volumen a visualizar.
    axs : array-like de matplotlib.axes.Axes, longitud 3
        Ejes de matplotlib donde se dibujarán los tres cortes.
    titulo : str, opcional
        Prefijo del título de cada subplot. Por defecto "Volumen".
    rot180 : bool, opcional
        Si es True, rota cada corte 180° antes de mostrarlo.
        Por defecto True.

    Retorna
    -------
    None
    """
    arr = sitk.GetArrayFromImage(vol_sitk)  # (Z,Y,X)
    z, y, x = arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2] // 2

    axial = arr[z, :, :]
    coronal = arr[:, y, :]
    sagittal = arr[:, :, x]

    if rot180:
        axial = np.rot90(axial, 2)
        coronal = np.rot90(coronal, 2)
        sagittal = np.rot90(sagittal, 2)

    axs[0].imshow(axial, cmap='gray')
    axs[0].set_title(f"{titulo} - Axial")
    axs[0].axis('off')

    axs[1].imshow(coronal, cmap='gray')
    axs[1].set_title(f"{titulo} - Coronal")
    axs[1].axis('off')

    axs[2].imshow(sagittal, cmap='gray')
    axs[2].set_title(f"{titulo} - Sagital")
    axs[2].axis('off')


# =========================
# === PCA (usada)
# =========================
def proyectar_plano(volume_sitk, plano="YZ", hu_umbral=300):
    """
    Estima el ángulo principal de orientación del cráneo en un plano dado.

    Binariza el volumen CT usando 'hu_umbral' (Unidades Hounsfield) para
    aislar estructuras densas (hueso), proyecta las coordenadas sobre el
    plano solicitado y aplica PCA para obtener el eje de mayor varianza.
    El ángulo retornado representa la inclinación del primer componente
    principal respecto al eje horizontal.

    Parámetros
    ----------
    volume_sitk : sitk.Image
        Volumen CT de entrada.
    plano : str, opcional
        Plano de proyección: "YZ", "XZ" o "XY". Por defecto "YZ".
    hu_umbral : int, opcional
        Umbral en HU para segmentar estructuras óseas. Por defecto 300.

    Retorna
    -------
    float
        Ángulo en grados del primer componente principal en el plano
        indicado, en el rango (-180°, 180°].

    Lanza
    -----
    ValueError
        Si 'plano' no es uno de los valores válidos ("YZ", "XZ", "XY").
    """
    array = sitk.GetArrayFromImage(volume_sitk)  # (Z,Y,X)
    binario = array > hu_umbral
    coords = np.argwhere(binario)

    if plano == "YZ":
        proj_coords = coords[:, [1, 0]]
    elif plano == "XZ":
        proj_coords = coords[:, [2, 0]]
    elif plano == "XY":
        proj_coords = coords[:, [2, 1]]
    else:
        raise ValueError("Plano no válido")

    pca = PCA(n_components=2)
    pca.fit(proj_coords)
    v = pca.components_[0]
    angulo = np.degrees(np.arctan2(v[1], v[0]))
    return angulo


# =========================
# === Matrices de rotación
# =========================
def obtener_rotacion_eje_x(angulo):
    """
    Construye la matriz de rotación 3D alrededor del eje X.

    Parámetros
    ----------
    angulo : float
        Ángulo de rotación en grados. Un valor positivo rota en sentido
        horario visto desde el eje X positivo (internamente t = -ángulo).

    Retorna
    -------
    np.ndarray, shape (3, 3)
        Matriz de rotación alrededor del eje X.
    """
    t = -np.radians(angulo)
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
    ])


def obtener_rotacion_eje_y(angulo):
    """
    Construye la matriz de rotación 3D alrededor del eje Y.

    Parámetros
    ----------
    angulo : float
        Ángulo de rotación en grados. Un valor positivo rota en sentido
        horario visto desde el eje Y positivo (internamente t = -ángulo).

    Retorna
    -------
    np.ndarray, shape (3, 3)
        Matriz de rotación alrededor del eje Y.
    """
    t = -np.radians(angulo)
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
    ])


def obtener_rotacion_eje_z(angulo):
    """
    Construye la matriz de rotación 3D alrededor del eje Z.

    Parámetros
    ----------
    angulo : float
        Ángulo de rotación en grados. Un valor positivo rota en sentido
        horario visto desde el eje Z positivo (internamente t = -ángulo).

    Retorna
    -------
    np.ndarray, shape (3, 3)
        Matriz de rotación alrededor del eje Z.
    """
    t = -np.radians(angulo)
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ])


# =========================
# === Guardar volumen
# =========================
def guardar_volumen_como_nii(volumen_sitk, salida_folder, nombre_archivo):
    """
    Guarda un volumen SimpleITK como archivo NIfTI comprimido (.nii.gz).

    Crea la carpeta de salida si no existe y escribe el volumen con el
    nombre estandarizado 'Volumen_reorientado_paciente_<nombre_archivo>.nii.gz'.

    Parámetros
    ----------
    volumen_sitk : sitk.Image
        Volumen a guardar.
    salida_folder : str
        Ruta a la carpeta donde se escribirá el archivo.
    nombre_archivo : str
        Identificador del paciente usado en el nombre del archivo de salida.

    Retorna
    -------
    None
    """
    if not os.path.exists(salida_folder):
        os.makedirs(salida_folder)

    nifti_filename = os.path.join(
        salida_folder, f"Volumen_reorientado_paciente_{nombre_archivo}.nii.gz"
    )

    sitk.WriteImage(volumen_sitk, nifti_filename)
    print(f"Volumen NIfTI guardado en: {nifti_filename}")