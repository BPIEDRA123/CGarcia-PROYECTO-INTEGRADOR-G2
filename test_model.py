# -*- coding: utf-8 -*-
"""
test_model.py - Pruebas unitarias para model.py
===============================================

Ejecuta con:
    pytest -q test_model.py

Requisitos:
    - pytest
    - numpy
    - scikit-learn
    - joblib
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Asegura import local
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import model as mdl
except Exception as e:
    raise RuntimeError(f"No se pudo importar model.py: {e}")


# ---------------------------------------------------------------------
# Fixtures y datos sintéticos
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_data():
    """Genera un dataset pequeño de arrays 2D normalizados en [0,1] y labels binarios.
    Dos patrones:
      - Clase 0: cuadrante superior izquierdo brillante.
      - Clase 1: gradiente horizontal.
    """
    rng = np.random.default_rng(123)
    n_per_class = 12
    H, W = 32, 32

    # Clase 0
    X0 = []
    for _ in range(n_per_class):
        a = np.zeros((H, W), dtype=np.float32)
        a[0:H//2, 0:W//2] = 1.0
        a += rng.normal(0, 0.02, size=a.shape).astype(np.float32)
        a = np.clip(a, 0.0, 1.0)
        X0.append(a)

    # Clase 1
    X1 = []
    base = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    for _ in range(n_per_class):
        a = base + rng.normal(0, 0.02, size=base.shape).astype(np.float32)
        a = np.clip(a, 0.0, 1.0)
        X1.append(a)

    X = X0 + X1
    y = np.array([0]*n_per_class + [1]*n_per_class, dtype=int)
    return X, y


# ---------------------------------------------------------------------
# Tests del featurizador y pipeline
# ---------------------------------------------------------------------

def test_featurizer_shape(toy_data):
    X, _ = toy_data
    fz = mdl.ImageStatsFeaturizer(bins=16, compute_sobel=True)
    F = fz.fit_transform(X)
    assert isinstance(F, np.ndarray)
    # 4 stats + 5 percentiles + 16 hist + 2 sobel = 27 + 16? -> Recuento correcto: 4 + 5 + 16 + 2 = 27
    assert F.shape[0] == len(X)
    assert F.shape[1] == 27

@pytest.mark.parametrize("clf", ["logreg", "svc", "rf"])
def test_build_pipeline_and_fit_predict(toy_data, clf):
    X, y = toy_data
    cfg = mdl.ModelConfig(classifier=clf, pca_components=None, bins=16, sobel=True)
    pipe = mdl.build_pipeline(cfg)
    pipe = mdl.fit_model(pipe, X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    # Debe aprender algo (accuracy > 0.7 en train para este dataset simple)
    acc = (preds == y).mean()
    assert acc > 0.7

def test_predict_proba_softmax_fallback(toy_data):
    X, y = toy_data
    # Usamos un clasificador que no necesariamente da predict_proba (por si acaso SVC sin probability)
    cfg = mdl.ModelConfig(classifier="svc", C=1.0, kernel="linear")
    pipe = mdl.build_pipeline(cfg)
    pipe = mdl.fit_model(pipe, X, y)

    proba = mdl.predict_proba(pipe, X)
    assert proba.shape[0] == len(X)
    # Suma por fila ~ 1.0
    row_sums = np.abs(proba.sum(axis=1) - 1.0)
    assert np.all(row_sums < 1e-5)

def test_evaluate_and_cv(toy_data):
    X, y = toy_data
    cfg = mdl.ModelConfig(classifier="logreg", bins=8, sobel=False)
    pipe = mdl.build_pipeline(cfg)
    pipe = mdl.fit_model(pipe, X, y)

    y_pred = pipe.predict(X)
    metrics = mdl.evaluate_model(y, y_pred, average="macro")
    for k in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
        assert k in metrics

    # Cross-validation rápida (3 folds)
    cv = mdl.cross_validate(pipe, X, y, folds=3, random_state=42)
    assert "f1_macro_mean" in cv and "f1_macro_std" in cv

def test_save_and_load_model(tmp_path: Path, toy_data):
    X, y = toy_data
    cfg = mdl.ModelConfig(classifier="rf", n_estimators=50, bins=8)
    pipe = mdl.build_pipeline(cfg)
    pipe = mdl.fit_model(pipe, X, y)

    path = tmp_path / "model.joblib"
    saved = mdl.save_model(pipe, path)
    assert saved.exists()

    loaded = mdl.load_model(saved)
    preds = loaded.predict(X)
    assert preds.shape == y.shape
    # Coherencia aproximada entre modelos guardado/cargado
    assert (preds == pipe.predict(X)).mean() > 0.95
