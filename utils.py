# -*- coding: utf-8 -*-
"""
utils.py — Funciones auxiliares
===============================
Funciones generales de apoyo para logging, métricas, archivos,
tiempos de ejecución, visualización y manejo de configuraciones.

Compatible con los módulos:
  - data_processing.py
  - model.py
  - train.py
  - evaluate.py

Fecha de generación: 2025-10-25 17:25:07
"""

from __future__ import annotations
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Logging y manejo de consola
# ------------------------------------------------------------------------------

def setup_logger(name: str = "project", log_file: Optional[Union[str, Path]] = None, level=logging.INFO) -> logging.Logger:
    """Crea un logger configurable con salida a archivo y consola."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    # Consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Archivo opcional
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------------------
# Temporización de procesos
# ------------------------------------------------------------------------------

class Timer:
    """Context manager simple para medir tiempos de ejecución."""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start_time
        msg = f"[{self.name}] Finalizado en {self.elapsed:.2f} s" if self.name else f"Tiempo: {self.elapsed:.2f} s"
        print(msg)


# ------------------------------------------------------------------------------
# Manejo de archivos y directorios
# ------------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Crea un directorio si no existe."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Guarda un diccionario como JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Carga un JSON a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_images(directory: Union[str, Path], extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    """Devuelve una lista de rutas de imágenes en un directorio."""
    return [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in extensions]


# ------------------------------------------------------------------------------
# Métricas y visualización
# ------------------------------------------------------------------------------

def plot_metrics(history: Dict[str, List[float]], out_path: Optional[Union[str, Path]] = None, title: str = "Training Metrics"):
    """Grafica métricas de entrenamiento (accuracy, loss, etc.) guardadas en un dict."""
    plt.figure(figsize=(6, 4))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if out_path:
        ensure_dir(Path(out_path).parent)
        plt.savefig(out_path, dpi=150)
    plt.close()

def summarize_metrics(metrics: Dict[str, Any]) -> str:
    """Devuelve una versión formateada de las métricas en texto."""
    lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items() if k != "confusion_matrix"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------------------

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Carga un archivo JSON de configuración (como el generado por train.py)."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------
# Utilidades adicionales
# ------------------------------------------------------------------------------

def describe_dataframe(df: pd.DataFrame, max_rows: int = 10) -> None:
    """Imprime un resumen del DataFrame."""
    print(f"Shape: {df.shape}")
    print("Columnas:", list(df.columns))
    print("Primeras filas:")
    print(df.head(max_rows))

def seed_everything(seed: int = 42):
    """Fija semillas globales para reproducibilidad."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


__all__ = [
    "setup_logger", "Timer",
    "ensure_dir", "save_json", "load_json", "list_images",
    "plot_metrics", "summarize_metrics",
    "load_config",
    "describe_dataframe", "seed_everything",
]
