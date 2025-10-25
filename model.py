# -*- coding: utf-8 -*-
"""
model.py — Definición del modelo
=================================
Módulo de definición y utilidades de modelo para clasificación de imágenes
(e.g., ecografías) usando un pipeline clásico de scikit-learn con
*featurización estadística* + clasificador.

Fecha de generación: 2025-10-25 17:17:12
Origen: Notebook "Integrador33.ipynb"

Notas:
  - Evitamos dependencias pesadas (TensorFlow/PyTorch). Si más adelante deseas
    una CNN, puedo proveer una versión 'torch' en un archivo aparte (p. ej. model_torch.py).
  - Este módulo funciona con arrays numpy provenientes de `data_processing.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Persistencia
import joblib
from pathlib import Path

# ------------------------------------------------------------------------------
# Featurizador: estadísticas simples + bordes
# ------------------------------------------------------------------------------

class ImageStatsFeaturizer(BaseEstimator, TransformerMixin):
    """Extrae un vector compacto de características por imagen.

    Features (por imagen):
      - mean, std, min, max (intensidad)
      - percentiles: p10, p25, p50, p75, p90
      - sobel magnitude stats: mean, std
      - histograma (N bins, por defecto 32, sobre [0,1])

    Espera arrays normalizados en [0,1] (usar `preprocess_image`).
    Si la imagen tiene 3 canales, la convierte a grises por media.
    """
    def __init__(self, bins: int = 32, compute_sobel: bool = True, random_state: int = 42):
        self.bins = bins
        self.compute_sobel = compute_sobel
        self.random_state = random_state

    def fit(self, X: Iterable[np.ndarray], y: Optional[np.ndarray] = None):
        return self

    def _to_gray(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr.mean(axis=-1)
        return arr

    def transform(self, X: Iterable[np.ndarray]) -> np.ndarray:
        feats: List[np.ndarray] = []
        for arr in X:
            a = self._to_gray(np.asarray(arr))
            a = a.astype(np.float32)
            # stats básicos
            vals = [
                float(np.mean(a)),
                float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
                float(np.min(a)),
                float(np.max(a)),
            ]
            # percentiles
            for p in (10, 25, 50, 75, 90):
                vals.append(float(np.percentile(a, p)))

            # histograma
            h, _ = np.histogram(a, bins=self.bins, range=(0.0, 1.0), density=True)
            vals.extend(h.astype(np.float32).tolist())

            # sobel opcional
            if self.compute_sobel:
                # sobel por ejes y magnitud
                sx = np.abs(np.gradient(a, axis=1))
                sy = np.abs(np.gradient(a, axis=0))
                mag = np.hypot(sx, sy)
                vals.extend([float(np.mean(mag)), float(np.std(mag, ddof=1)) if mag.size > 1 else 0.0])

            feats.append(np.array(vals, dtype=np.float32))

        return np.vstack(feats)

# ------------------------------------------------------------------------------
# Configuración y construcción del modelo
# ------------------------------------------------------------------------------

@dataclass
class ModelConfig:
    classifier: str = "logreg"   # 'logreg' | 'svc' | 'rf'
    pca_components: Optional[int] = None
    # Hiperparámetros por modelo
    C: float = 1.0               # logreg/svc
    kernel: str = "rbf"          # svc
    n_estimators: int = 200      # rf
    max_depth: Optional[int] = None
    random_state: int = 42
    sobel: bool = True
    bins: int = 32

def _make_classifier(cfg: ModelConfig):
    if cfg.classifier == "logreg":
        return LogisticRegression(
            C=cfg.C, max_iter=2000, class_weight="balanced", n_jobs=None, random_state=cfg.random_state
        )
    elif cfg.classifier == "svc":
        return SVC(C=cfg.C, kernel=cfg.kernel, probability=True, class_weight="balanced", random_state=cfg.random_state)
    elif cfg.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
            random_state=cfg.random_state, class_weight="balanced_subsample", n_jobs=-1
        )
    else:
        raise ValueError("classifier debe ser 'logreg', 'svc' o 'rf'.")

def build_pipeline(cfg: Optional[ModelConfig] = None) -> Pipeline:
    """Construye un Pipeline sklearn: Featurizer -> (PCA?) -> Scaler -> Clasificador"""
    cfg = cfg or ModelConfig()
    steps = []
    steps.append(("featurizer", ImageStatsFeaturizer(bins=cfg.bins, compute_sobel=cfg.sobel, random_state=cfg.random_state)))
    if cfg.pca_components:
        steps.append(("pca", PCA(n_components=cfg.pca_components, random_state=cfg.random_state)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("clf", _make_classifier(cfg)))
    return Pipeline(steps)

# ------------------------------------------------------------------------------
# Entrenamiento, evaluación y persistencia
# ------------------------------------------------------------------------------

def fit_model(pipeline: Pipeline, X: List[np.ndarray], y: np.ndarray) -> Pipeline:
    """Entrena el pipeline con X (arrays) e y (labels)."""
    pipeline.fit(X, y)
    return pipeline

def predict_proba(pipeline: Pipeline, X: List[np.ndarray]) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    # Fallback para clasificadores sin proba
    preds = pipeline.decision_function(X)
    # convertir a pseudo-probabilidades usando softmax
    if preds.ndim == 1:
        preds = np.vstack([-preds, preds]).T
    e = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

def cross_validate(pipeline: Pipeline, X: List[np.ndarray], y: np.ndarray, folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return {"f1_macro_mean": float(scores.mean()), "f1_macro_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0}

def save_model(pipeline: Pipeline, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path

def load_model(path: Union[str, Path]) -> Pipeline:
    return joblib.load(path)

__all__ = [
    "ImageStatsFeaturizer",
    "ModelConfig",
    "build_pipeline",
    "fit_model",
    "predict_proba",
    "evaluate_model",
    "cross_validate",
    "save_model",
    "load_model",
]
