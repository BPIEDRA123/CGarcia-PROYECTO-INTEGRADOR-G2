# -*- coding: utf-8 -*-
"""
train.py — Script de entrenamiento
==================================
Entrena un clasificador clásico (scikit-learn) usando el pipeline definido en model.py
y las utilidades de preprocesamiento en data_processing.py.

Flujo:
  1) Construye un DataFrame desde un directorio etiquetado (carpeta = label).
  2) Split train/val/test.
  3) Preprocesa imágenes (resize + normaliza).
  4) Construye y entrena el pipeline (featurizador + (PCA?) + scaler + clasificador).
  5) Evalúa en val/test, guarda métricas, predicciones y el modelo.

Ejemplo:
  python train.py \
    --data_dir /ruta/dataset \
    --output_dir ./outputs/exp1 \
    --classifier svc --C 2.0 --kernel rbf --pca_components 32

Fecha de generación: 2025-10-25 17:19:22
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Módulos locales
try:
    from data_processing import (
        build_dataframe_from_dir,
        split_dataframe,
        preprocess_image,
        load_image,
    )
    from model import (
        ModelConfig,
        build_pipeline,
        fit_model,
        predict_proba,
        evaluate_model,
        cross_validate,
        save_model,
    )
except Exception as e:
    print("Error importando módulos locales (data_processing, model).", file=sys.stderr)
    raise

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento de clasificador (scikit-learn).")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio raíz del dataset (carpetas = labels).")
    p.add_argument("--output_dir", type=str, default="./outputs/exp", help="Directorio de salida para artefactos.")
    # Preprocesamiento
    p.add_argument("--img_size", type=str, default="224,224", help="Tamaño de imagen WxH (por defecto 224,224).")
    p.add_argument("--img_mode", type=str, default="L", choices=["L","RGB"], help="Modo de imagen PIL ('L' o 'RGB').")
    # Splits
    p.add_argument("--test_size", type=float, default=0.2, help="Proporción test.")
    p.add_argument("--val_size", type=float, default=0.1, help="Proporción val (del resto).")
    p.add_argument("--no_stratify", action="store_true", help="Desactiva estratificación por label.")
    p.add_argument("--random_state", type=int, default=42, help="Seed.")
    # Modelo
    p.add_argument("--classifier", type=str, default="logreg", choices=["logreg","svc","rf"], help="Clasificador.")
    p.add_argument("--pca_components", type=int, default=None, help="Componentes PCA (opcional).")
    p.add_argument("--C", type=float, default=1.0, help="C para logreg/svc.")
    p.add_argument("--kernel", type=str, default="rbf", help="Kernel SVC.")
    p.add_argument("--n_estimators", type=int, default=200, help="Árboles para RandomForest.")
    p.add_argument("--max_depth", type=int, default=None, help="Profundidad máxima para RandomForest.")
    p.add_argument("--bins", type=int, default=32, help="Bins del histograma del featurizador.")
    p.add_argument("--no_sobel", action="store_true", help="Desactiva estadísticas Sobel en featurizador.")
    # Validación
    p.add_argument("--cv_folds", type=int, default=0, help="Folds para CV (0 = desactivado).")
    return p.parse_args()

def _parse_size(s: str) -> Tuple[int,int]:
    try:
        w, h = s.split(",")
        return int(w), int(h)
    except Exception:
        raise ValueError("img_size debe ser 'W,H', p.ej. '224,224'.")

def _preprocess_batch(paths: List[str], mode: str, size: Tuple[int,int]) -> List[np.ndarray]:
    X: List[np.ndarray] = []
    for p in paths:
        arr = preprocess_image(load_image(p, mode=mode), size=size, keep_aspect=True, normalize=True)
        X.append(arr)
    return X

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = _parse_size(args.img_size)
    stratify = not args.no_stratify

    print("=== Paso 1: Construyendo DataFrame desde directorio ===")
    df = build_dataframe_from_dir(args.data_dir)
    if df.empty:
        print("No se encontraron imágenes en:", args.data_dir, file=sys.stderr)
        sys.exit(1)
    df.to_csv(out_dir / "dataset_index.csv", index=False)
    print("Total imágenes:", len(df), " | Clases:", df["label"].nunique())

    print("=== Paso 2: Generando splits train/val/test ===")
    splits = split_dataframe(df, test_size=args.test_size, val_size=args.val_size, stratify=stratify, random_state=args.random_state)
    for k, d in splits.items():
        d.to_csv(out_dir / f"split_{{k}}.csv", index=False)
        print(f"Split {{k}}: {{len(d)}}")

    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    print("=== Paso 3: Codificando labels ===")
    le = LabelEncoder()
    le.fit(train_df["label"])
    np.save(out_dir / "label_classes.npy", le.classes_)

    print("=== Paso 4: Preprocesando imágenes (train/val/test) ===")
    X_train = _preprocess_batch(train_df["filepath"].tolist(), args.img_mode, img_size)
    y_train = le.transform(train_df["label"].tolist())

    X_val = _preprocess_batch(val_df["filepath"].tolist(), args.img_mode, img_size)
    y_val = le.transform(val_df["label"].tolist())

    X_test = _preprocess_batch(test_df["filepath"].tolist(), args.img_mode, img_size)
    y_test = le.transform(test_df["label"].tolist())

    print("=== Paso 5: Construyendo pipeline ===")
    cfg = ModelConfig(
        classifier=args.classifier,
        pca_components=args.pca_components,
        C=args.C, kernel=args.kernel,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        random_state=args.random_state,
        sobel=(not args.no_sobel),
        bins=args.bins,
    )
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, ensure_ascii=False)

    pipe = build_pipeline(cfg)

    if args.cv_folds and args.cv_folds > 1:
        print(f"=== Paso 6 (opcional): Cross-Validation ({{args.cv_folds}} folds) ===")
        cv_res = cross_validate(pipe, X_train, y_train, folds=args.cv_folds, random_state=args.random_state)
        with open(out_dir / "cv_results.json", "w", encoding="utf-8") as f:
            json.dump(cv_res, f, indent=2, ensure_ascii=False)
        print("CV:", cv_res)

    print("=== Paso 7: Entrenando en TRAIN ===")
    pipe = fit_model(pipe, X_train, y_train)

    print("=== Paso 8: Evaluando en VALIDATION ===")
    y_val_pred = pipe.predict(X_val)
    val_metrics = evaluate_model(y_val, y_val_pred)
    with open(out_dir / "metrics_val.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)
    print("VAL metrics:", val_metrics)

    print("=== Paso 9: Retrain en TRAIN+VAL y evaluación final en TEST ===")
    X_tv = X_train + X_val
    y_tv = np.concatenate([y_train, y_val], axis=0)
    pipe = fit_model(pipe, X_tv, y_tv)

    # Guardar modelo antes de test
    model_path = out_dir / "model.joblib"
    save_model(pipe, model_path)

    y_test_pred = pipe.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred)
    with open(out_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print("TEST metrics:", test_metrics)

    # Predicciones + probabilidades
    try:
        proba = predict_proba(pipe, X_test)
        conf = np.max(proba, axis=1).tolist()
    except Exception:
        proba = None
        conf = [None] * len(y_test_pred)

    preds_df = pd.DataFrame({
        "filepath": test_df["filepath"].tolist(),
        "true_label": le.inverse_transform(y_test).tolist(),
        "pred_label": le.inverse_transform(y_test_pred).tolist(),
        "confidence": conf,
    })
    preds_df.to_csv(out_dir / "predictions_test.csv", index=False)

    print("=== Entrenamiento finalizado ===")
    print("Artefactos guardados en:", out_dir)

if __name__ == "__main__":
    main()
