# -*- coding: utf-8 -*-
"""
test_data_processing.py - Pruebas unitarias para data_processing.py
==================================================================

Ejecuta con:
    pytest -q test_data_processing.py

Requisitos:
    - pytest
    - pillow
    - numpy
    - pandas
    - scikit-learn
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Asegura que el módulo local sea importable (misma carpeta o raíz del proyecto)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import data_processing as dp
except Exception as e:
    raise RuntimeError(f"No se pudo importar data_processing.py: {e}")

import pytest


# ---------------------------------------------------------------------
# Fixtures de utilidad
# ---------------------------------------------------------------------

@pytest.fixture
def tmp_images_dir(tmp_path: Path) -> Path:
    """Crea un arbol de directorios con imagenes sinteticas etiquetadas."""
    root = tmp_path / "dataset"
    a = root / "class_a"
    b = root / "class_b"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)

    # class_a: patron con esquina superior izquierda brillante
    for i in range(3):
        arr = np.zeros((16, 16), dtype=np.uint8)
        arr[0:8, 0:8] = 255  # cuadrante brillante
        Image.fromarray(arr, mode="L").save(a / f"a_{i}.png")

    # class_b: gradiente horizontal
    for i in range(3):
        x = np.linspace(0, 255, 16, dtype=np.uint8)
        arr = np.tile(x, (16, 1))
        Image.fromarray(arr, mode="L").save(b / f"b_{i}.png")

    return root


# ---------------------------------------------------------------------
# Tests de I/O y preprocesamiento basico
# ---------------------------------------------------------------------

def test_build_dataframe_from_dir(tmp_images_dir: Path):
    df = dp.build_dataframe_from_dir(tmp_images_dir)
    assert not df.empty
    assert set(["filepath", "filename", "label"]).issubset(df.columns)
    # 6 imagenes en total
    assert len(df) == 6
    # Labels detectados
    assert set(df["label"].unique()) == {"class_a", "class_b"}

def test_load_and_preprocess_roundtrip(tmp_images_dir: Path):
    df = dp.build_dataframe_from_dir(tmp_images_dir)
    p = df.iloc[0]["filepath"]
    img = dp.load_image(p, mode="L")
    arr = dp.preprocess_image(img, size=(32, 32), keep_aspect=True, normalize=True)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (32, 32)
    assert 0.0 <= arr.min() <= 1.0 and 0.0 <= arr.max() <= 1.0

    # Roundtrip a imagen
    img2 = dp.array_to_image(arr)
    assert isinstance(img2, Image.Image)
    assert img2.size == (32, 32)

def test_gaussian_blur_array_delta():
    # Imagen delta (un pixel en 1.0)
    arr = np.zeros((21, 21), dtype=np.float32)
    arr[10, 10] = 1.0
    blurred = dp.gaussian_blur_array(arr, sigma=2.0)
    # El pico debe disminuir
    assert blurred.max() < arr.max()
    # La energia total debe conservarse aproximadamente (margen)
    assert np.isclose(blurred.sum(), arr.sum(), atol=1e-2)

def test_sobel_edges_nonzero():
    # Gradiente simple horizontal
    x = np.linspace(0, 1, 32, dtype=np.float32)
    arr = np.tile(x, (32, 1))
    edges = dp.sobel_edges(arr, axis=None)
    assert edges.shape == arr.shape
    assert edges.max() > 0.0

def test_augment_image_flips():
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[0, 0] = 255
    img = Image.fromarray(arr, mode="L")
    img_h = dp.augment_image(img, hflip=True)
    img_v = dp.augment_image(img, vflip=True)

    a_h = np.array(img_h)
    a_v = np.array(img_v)
    # Despues de flip horizontal, el 255 debe estar en (0, -1)
    assert a_h[0, -1] == 255
    # Despues de flip vertical, el 255 debe estar en (-1, 0)
    assert a_v[-1, 0] == 255


# ---------------------------------------------------------------------
# Tests de utilidades de dataset y estadisticas
# ---------------------------------------------------------------------

def test_compute_dataset_stats(tmp_images_dir: Path):
    df = dp.build_dataframe_from_dir(tmp_images_dir)
    stats = dp.compute_dataset_stats(df, sample_size=4, mode="L", size=(32, 32))
    assert "class_counts" in stats
    assert "pixel_mean_std" in stats
    counts = stats["class_counts"]
    assert counts["class_a"] == 3 and counts["class_b"] == 3
    mean, std = stats["pixel_mean_std"]
    assert 0.0 <= mean <= 1.0
    assert std >= 0.0

def test_split_dataframe_stratified(tmp_images_dir: Path):
    df = dp.build_dataframe_from_dir(tmp_images_dir)
    splits = dp.split_dataframe(df, test_size=0.33, val_size=0.17, stratify=True, random_state=42)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Total consistente
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Todas las clases presentes en cada split (dataset pequeno pero balanceado)
    for part in (train_df, val_df, test_df):
        assert set(part["label"].unique()) == {"class_a", "class_b"}

def test_save_and_process_pipeline(tmp_images_dir: Path, tmp_path: Path):
    # Toma una imagen y ejecuta el pipeline completo process_and_save
    df = dp.build_dataframe_from_dir(tmp_images_dir)
    p = df.iloc[0]["filepath"]
    out_file = tmp_path / "out.png"
    result = dp.process_and_save(p, out_file, size=(64, 64), do_sobel=True, blur_sigma=1.0, mode="L")
    assert result.exists()

    # save_processed_array directo
    arr = np.ones((32, 32), dtype=np.float32)
    out2 = tmp_path / "out2.png"
    dp.save_processed_array(arr, out2)
    assert out2.exists()


# ---------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------

def test_set_seed_reproducible():
    dp.set_seed(123)
    a = np.random.rand(5)
    dp.set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)
