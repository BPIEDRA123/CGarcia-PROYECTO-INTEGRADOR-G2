# -*- coding: utf-8 -*-
"""
evaluate.py — Script de evaluación
==================================
Carga un modelo entrenado (joblib) y evalúa sobre un conjunto de imágenes
usando las mismas utilidades de preprocesamiento.

Fuentes posibles:
  A) Un CSV de split (p. ej. split_test.csv) con columnas: filepath,label
  B) Un directorio etiquetado (carpetas = labels)

Genera:
  - metrics_eval.json (accuracy, precision, recall, f1, matriz de confusión)
  - predictions_eval.csv (filepath, true_label, pred_label, confidence)
  - classification_report.txt
  - confusion_matrix.png (opcional con --plot_cm)
  - roc_pr_curves.png (opcional binario con --plot_curves)

Fecha de generación: 2025-10-25 17:22:21
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

# Importa módulos locales
try:
    from data_processing import (
        build_dataframe_from_dir,
        preprocess_image,
        load_image,
    )
    from model import (
        load_model,
        predict_proba,
    )
except Exception as e:
    print("Error importando módulos locales (data_processing, model).", file=sys.stderr)
    raise

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación de un modelo (scikit-learn).")
    p.add_argument("--model_path", type=str, required=True, help="Ruta al modelo .joblib.")
    p.add_argument("--labels_path", type=str, required=True, help="Ruta a label_classes.npy (orden de clases).")
    p.add_argument("--output_dir", type=str, default="./outputs/eval", help="Directorio de salida.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split_csv", type=str, help="CSV con columnas filepath,label (p. ej., split_test.csv).")
    src.add_argument("--data_dir", type=str, help="Directorio etiquetado (carpetas = labels).")
    p.add_argument("--img_size", type=str, default="224,224", help="Tamaño imagen WxH (ej. 224,224).")
    p.add_argument("--img_mode", type=str, default="L", choices=["L","RGB"], help="Modo imagen PIL.")
    # Plots opcionales
    p.add_argument("--plot_cm", action="store_true", help="Guardar matriz de confusión como imagen.")
    p.add_argument("--plot_curves", action="store_true", help="Guardar curvas ROC/PR (solo binario).")
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

# ------------------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------------------

def evaluate(y_true_ids: np.ndarray, y_pred_ids: np.ndarray, class_names: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true_ids, y_pred_ids))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_ids, y_pred_ids, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_ids, y_pred_ids)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classes": class_names.tolist(),
    }

def plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    # valores
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_curves_binary(y_true_ids: np.ndarray, proba: np.ndarray, class_names: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    # Asume 2 clases, toma probabilidad de la clase positiva (índice 1)
    y_true = (y_true_ids == 1).astype(int)
    y_score = proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)

    # ROC
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(fpr, tpr)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve (AUC = {:.3f})".format(roc_auc))
    fig1.tight_layout()
    fig1.savefig(out_path.parent / "roc_curve.png", dpi=150)
    plt.close(fig1)

    # PR
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(rec, prec)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve (AUC = {:.3f})".format(pr_auc))
    fig2.tight_layout()
    fig2.savefig(out_path.parent / "pr_curve.png", dpi=150)
    plt.close(fig2)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = _parse_size(args.img_size)

    # Cargar clases y modelo
    class_names = np.load(args.labels_path, allow_pickle=True)
    model = load_model(args.model_path)

    # Armar dataframe fuente
    if args.split_csv:
        df = pd.read_csv(args.split_csv)
        if not {"filepath", "label"}.issubset(df.columns):
            print("El CSV debe contener columnas: filepath,label", file=sys.stderr)
            sys.exit(1)
    else:
        df = build_dataframe_from_dir(args.data_dir)

    if df.empty:
        print("No hay datos para evaluar.", file=sys.stderr)
        sys.exit(1)

    # Preprocesar
    X = _preprocess_batch(df["filepath"].tolist(), args.img_mode, img_size)

    # Mapear labels string -> ids según class_names
    label_to_id = {name: i for i, name in enumerate(class_names)}
    try:
        y_true_ids = np.array([label_to_id[lbl] for lbl in df["label"].tolist()], dtype=int)
    except KeyError as e:
        print(f"Etiqueta no encontrada en label_classes.npy: {e}", file=sys.stderr)
        sys.exit(1)

    # Predicción
    y_pred_ids = model.predict(X)

    # Métricas
    metrics = evaluate(y_true_ids, y_pred_ids, class_names)
    with open(out_dir / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Reporte de clasificación por clase
    report_txt = classification_report(y_true_ids, y_pred_ids, target_names=class_names, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # Guardar predicciones y confidencias
    try:
        proba = predict_proba(model, X)
        conf = np.max(proba, axis=1).tolist()
    except Exception:
        proba = None
        conf = [None] * len(y_pred_ids)

    preds_df = pd.DataFrame({
        "filepath": df["filepath"].tolist(),
        "true_label": [class_names[i] for i in y_true_ids],
        "pred_label": [class_names[i] for i in y_pred_ids],
        "confidence": conf,
    })
    preds_df.to_csv(out_dir / "predictions_eval.csv", index=False)

    # Plots opcionales
    if args.plot_cm:
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    if args.plot_curves and proba is not None and len(class_names) == 2:
        plot_curves_binary(y_true_ids, proba, class_names, out_dir / "roc_pr_curves.png")

    print("Evaluación completa. Artefactos guardados en:", out_dir)

if __name__ == "__main__":
    main()
