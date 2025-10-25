# -*- coding: utf-8 -*-
"""
data_processing.py — Funciones de procesamiento de datos/imagenes
=================================================================
Módulo utilitario generado a partir de la revisión del notebook "Integrador33.ipynb"
(fecha de generación: 2025-10-25 17:14:51).

En el notebook se detectaron librerías de procesamiento como:
  - numpy, pandas
  - PIL (Image), scipy.ndimage (sobel, gaussian_filter)
  - matplotlib, seaborn

Este módulo expone funciones reutilizables para:
  * Carga y preprocesamiento de imágenes (escala de grises, resize, normalización).
  * Filtros básicos (gaussian blur) y detección de bordes (Sobel).
  * Aumento de datos (rotación, flips, brillo/contraste).
  * Utilidades para construir un DataFrame de un directorio etiquetado.
  * Estadísticos de dataset y partición train/val/test.

Ajusta las funciones a tu proyecto según los requerimientos específicos.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd

from PIL import Image, ImageEnhance
from scipy.ndimage import sobel, gaussian_filter
from sklearn.model_selection import train_test_split
import warnings

# ------------------------------------------------------------------------------
# Configuración y utilidades generales
# ------------------------------------------------------------------------------

Image.MAX_IMAGE_PIXELS = None  # Evita warnings con imágenes grandes

def set_seed(seed: int = 42) -> None:
    """
    Fija seeds para reproducibilidad (numpy).
    """
    np.random.seed(seed)

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Crea el directorio si no existe y retorna Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------------------------------------------------------------------
# Carga y conversión de imágenes
# ------------------------------------------------------------------------------

def load_image(path: Union[str, Path], mode: str = "L") -> Image.Image:
    """
    Carga una imagen desde disco.

    Args:
        path: Ruta al archivo.
        mode: Modo PIL para convertir. "L"=grises, "RGB"=color.

    Returns:
        PIL.Image.Image
    """
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return img

def image_to_array(img: Image.Image, normalize: bool = True) -> np.ndarray:
    """
    Convierte PIL.Image a np.ndarray (C*H*W si es necesario).

    Args:
        img: Imagen PIL.
        normalize: Si True, escala a [0,1] float32.

    Returns:
        Array numpy HxW o HxWxC.
    """
    arr = np.array(img)
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    return arr

def array_to_image(arr: np.ndarray) -> Image.Image:
    """
    Convierte array a imagen PIL (revirtiendo normalización si aplica).
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)

# ------------------------------------------------------------------------------
# Preprocesamiento y filtros
# ------------------------------------------------------------------------------

def preprocess_image(
    img: Image.Image,
    size: Tuple[int, int] = (224, 224),
    keep_aspect: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Redimensiona (conservar aspecto opcional) y normaliza.

    Args:
        img: Imagen PIL.
        size: (ancho, alto) de salida.
        keep_aspect: Si True, hace letterbox manteniendo aspecto.
        normalize: Escala a [0,1].

    Returns:
        np.ndarray con la imagen preprocesada.
    """
    if keep_aspect:
        img = _resize_letterbox(img, size)
    else:
        img = img.resize(size)
    return image_to_array(img, normalize=normalize)

def _resize_letterbox(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Redimensiona manteniendo aspecto y añade bordes (letterbox) si es necesario.
    """
    target_w, target_h = size
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = img.resize((new_w, new_h))
    canvas = Image.new(img.mode, (target_w, target_h), 0 if img.mode == "L" else (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas

def gaussian_blur_array(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Aplica blur gaussiano a un array (ya normalizado o no).

    Args:
        arr: Imagen en array (HxW o HxWxC).
        sigma: Desviación estándar del kernel gaussiano.

    Returns:
        Array filtrado.
    """
    if arr.ndim == 2:
        return gaussian_filter(arr, sigma=sigma)
    elif arr.ndim == 3:
        # Aplica por canal
        return np.stack([gaussian_filter(arr[..., c], sigma=sigma) for c in range(arr.shape[-1])], axis=-1)
    else:
        raise ValueError("Array con dimensiones no soportadas.")

def sobel_edges(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Detección de bordes con operador Sobel.

    Args:
        arr: Imagen en array (grises o RGB).
        axis: None (magnitud), 0 (vertical), 1 (horizontal).

    Returns:
        Array con bordes.
    """
    if arr.ndim == 3:
        # Convertir a grises simple si RGB
        arr = arr.mean(axis=-1)

    if axis is None:
        sx = sobel(arr, axis=1)
        sy = sobel(arr, axis=0)
        mag = np.hypot(sx, sy)
        mag = (mag / (mag.max() + 1e-8)).astype(np.float32)
        return mag
    elif axis in (0, 1):
        s = sobel(arr, axis=axis)
        s = (np.abs(s) / (np.max(np.abs(s)) + 1e-8)).astype(np.float32)
        return s
    else:
        raise ValueError("axis debe ser None, 0 o 1.")

def adjust_brightness_contrast(
    img: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    """
    Ajusta brillo y contraste. 1.0 = sin cambio.
    """
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

# ------------------------------------------------------------------------------
# Aumento de datos
# ------------------------------------------------------------------------------

def augment_image(
    img: Image.Image,
    rotate_deg: float = 0.0,
    hflip: bool = False,
    vflip: bool = False,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    """
    Aplica transformaciones simples para aumentar diversidad.
    """
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True, fillcolor=0 if img.mode == "L" else (0, 0, 0))
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = adjust_brightness_contrast(img, brightness=brightness, contrast=contrast)
    return img

# ------------------------------------------------------------------------------
# Dataset utilities (directorios -> DataFrame)
# ------------------------------------------------------------------------------

def build_dataframe_from_dir(
    root_dir: Union[str, Path],
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    label_from_parent: bool = True,
) -> pd.DataFrame:
    """
    Recorre un directorio y crea un DataFrame con columnas:
      - filepath
      - label (si label_from_parent=True, usa el nombre de la carpeta padre)
      - filename

    Estructura esperada si label_from_parent:
        root/
          clase_a/ img1.png, img2.png
          clase_b/ img3.png, img4.png
    """
    root = Path(root_dir)
    paths: List[Path] = []
    labels: List[str] = []
    filenames: List[str] = []

    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(p)
            filenames.append(p.name)
            if label_from_parent:
                labels.append(p.parent.name)
            else:
                labels.append("")

    df = pd.DataFrame({
        "filepath": [str(p) for p in paths],
        "filename": filenames,
        "label": labels,
    })
    return df

def compute_dataset_stats(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    mode: str = "L",
    size: Tuple[int, int] = (224, 224),
) -> Dict[str, Union[Dict[str, int], Tuple[float, float]]]:
    """
    Calcula conteo por clase y media/desviación (pixel) aproximada.

    Nota: para media/std, se usa una muestra (opcional) para eficiencia.
    """
    out: Dict[str, Union[Dict[str, int], Tuple[float, float]]] = {}

    if "label" in df.columns:
        out["class_counts"] = df["label"].value_counts().to_dict()  # type: ignore[assignment]

    paths = df["filepath"].tolist()
    if sample_size:
        rng = np.random.default_rng(42)
        # sin reemplazo
        if sample_size < len(paths):
            idx = rng.choice(len(paths), size=sample_size, replace=False).tolist()
            paths = [paths[i] for i in idx]

    accum = []
    for p in paths:
        try:
            arr = preprocess_image(load_image(p, mode=mode), size=size, keep_aspect=True, normalize=True)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)  # gris
            accum.append(arr)
        except Exception as e:
            warnings.warn(f"No se pudo procesar {p}: {e}")

    if accum:
        stack = np.stack(accum)
        mean = float(stack.mean())
        std = float(stack.std(ddof=1)) if stack.size > 1 else 0.0
        out["pixel_mean_std"] = (mean, std)

    return out

# ------------------------------------------------------------------------------
# Particiones y guardado
# ------------------------------------------------------------------------------

def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Separa un DataFrame (con columna 'label') en splits train/val/test.

    Args:
        df: DataFrame con al menos columna 'filepath' y opcional 'label'.
        test_size: proporción para test.
        val_size: proporción para validación (aplicada sobre el resto tras test).
        stratify: si True, usa la columna 'label' para estratificar.
    """
    if stratify and "label" in df.columns and df["label"].nunique() > 1:
        strat = df["label"]
    else:
        strat = None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    if stratify and "label" in train_df.columns and train_df["label"].nunique() > 1:
        strat2 = train_df["label"]
    else:
        strat2 = None

    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=random_state, stratify=strat2
    )
    return {{
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }}

def save_processed_array(
    arr: np.ndarray,
    out_path: Union[str, Path],
) -> None:
    """
    Guarda un array (normalizado [0,1] o [0,255]) como imagen en disco.
    """
    img = array_to_image(arr)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    img.save(out_path)

# ------------------------------------------------------------------------------
# Pipeline de ejemplo
# ------------------------------------------------------------------------------

def process_and_save(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    size: Tuple[int, int] = (224, 224),
    do_sobel: bool = False,
    blur_sigma: Optional[float] = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    mode: str = "L",
) -> Path:
    """
    Pipeline simple: carga -> augment -> resize -> filtros -> guarda.

    Returns:
        Ruta de salida como Path.
    """
    img = load_image(in_path, mode=mode)
    img = adjust_brightness_contrast(img, brightness=brightness, contrast=contrast)
    arr = preprocess_image(img, size=size, keep_aspect=True, normalize=True)

    if blur_sigma is not None:
        arr = gaussian_blur_array(arr, sigma=blur_sigma)

    if do_sobel:
        arr = sobel_edges(arr, axis=None)

    save_processed_array(arr, out_path)
    return Path(out_path)

__all__ = [
    # Generales
    "set_seed", "ensure_dir",
    # I/O y conversiones
    "load_image", "image_to_array", "array_to_image",
    # Preprocesamiento y filtros
    "preprocess_image", "gaussian_blur_array", "sobel_edges",
    "adjust_brightness_contrast", "augment_image",
    # Dataset utils
    "build_dataframe_from_dir", "compute_dataset_stats",
    # Splits y guardado
    "split_dataframe", "save_processed_array",
    # Pipeline
    "process_and_save",
]
