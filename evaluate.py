"""
evaluate.py — Script de evaluación

Generado a partir de Integrador31.ipynb.
Evalúa un modelo sobre el split de test usando las utilidades de `data_processing` y `model`.
- Carga modelo (SavedModel/.h5) o lo reconstruye si no se provee `--model_path`.
- Reporta métricas (`model.evaluate`) y, opcionalmente, classification report y matriz de confusión.
- Puede guardar un JSON con resultados vía `--save_report`.
"""

from __future__ import annotations

def _load_keras_model(model_path: str):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"[evaluate] No se pudo cargar el modelo Keras desde '{model_path}': {e}")
        return None

def evaluate_model(
    base_path: str,
    model_path: str | None = None,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    max_images_per_class: int | None = None,
    classification_report_flag: bool = True,
    confusion_matrix_flag: bool = True,
):
    """
    Retorna (metrics_dict, reports_dict). `reports_dict` puede incluir:
      - "classification_report" (str)
      - "confusion_matrix" (list[list[int]])
    """
    from data_processing import cargar_dataset_completo_avanzado
    from model import crear_modelo_prediccion_compatible

    # 1) Dataset de test
    ds = cargar_dataset_completo_avanzado(
        base_path=base_path,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        test_split=test_split,
        max_images_per_class=max_images_per_class,
    )
    test_data = ds.get("test")
    if test_data is None:
        raise RuntimeError("No se encontró split de test. Ajusta `--test_split` o prepara el conjunto de test.")

    # 2) Inferir num_classes e input_shape
    try:
        from data_processing import CLASSES
        num_classes = len(CLASSES)
    except Exception:
        num_classes = getattr(test_data, "num_classes", None) or 2
    input_shape = (img_size[0], img_size[1], 3)

    # 3) Cargar o crear modelo
    model = None
    if model_path:
        model = _load_keras_model(model_path)
    if model is None:
        model = crear_modelo_prediccion_compatible(input_shape=input_shape, num_classes=num_classes)

    # 4) Métricas con .evaluate()
    metrics = {}
    if hasattr(model, "evaluate"):
        try:
            res = model.evaluate(test_data, return_dict=True)
            if isinstance(res, dict):
                metrics.update(res)
        except TypeError:
            res = model.evaluate(test_data)
            if isinstance(res, (list, tuple)) and res:
                metrics["test_loss"] = res[0]
            else:
                metrics["test_metric"] = float(res)

    # 5) Predicciones para reportes detallados
    reports = {}
    if classification_report_flag or confusion_matrix_flag:
        try:
            import numpy as np
            y_true_all, y_pred_all = [], []
            # Intento genérico de iterar test_data como (x, y)
            for batch in test_data:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    bx, by = batch[0], batch[1]
                    if hasattr(model, "predict"):
                        py = model.predict(bx, verbose=0)
                        y_pred_all.append(py)
                        y_true_all.append(by)
            if y_true_all and y_pred_all:
                y_true = np.concatenate(y_true_all, axis=0)
                y_pred = np.concatenate(y_pred_all, axis=0)
                # Convertir a clases
                if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
                    y_pred_cls = np.argmax(y_pred, axis=-1)
                else:
                    y_pred_cls = (y_pred.ravel() >= 0.5).astype(int)
                # y_true a clases
                try:
                    y_true_cls = y_true.argmax(axis=-1) if y_true.ndim > 1 and y_true.shape[-1] > 1 else y_true.ravel().astype(int)
                except Exception:
                    y_true_cls = y_true

                from sklearn.metrics import classification_report, confusion_matrix
                if classification_report_flag:
                    reports["classification_report"] = classification_report(y_true_cls, y_pred_cls, zero_division=0)
                if confusion_matrix_flag:
                    reports["confusion_matrix"] = confusion_matrix(y_true_cls, y_pred_cls).tolist()
        except Exception as e:
            print(f"[evaluate] No se pudieron generar reportes detallados: {e}")

    return metrics, reports

# ---- Utilidades de evaluación extraídas del notebook ----
def cargar_y_preprocesar_imagen_avanzado(ruta, tamaño=IMG_SIZE):
    try:
        with Image.open(ruta) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = mejorar_calidad_imagen(img)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.SMOOTH)
            img.thumbnail((tamaño[0] * 2, tamaño[1] * 2), Image.Resampling.LANCZOS)
            img = img.resize(tamaño, Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.power(arr, 1.1)
            if arr.shape != (*tamaño, 3):
                import cv2
                arr = cv2.resize(arr, tamaño)
            return arr
    except Exception as e:
        print(f"❌ Error avanzado procesando {ruta}: {e}")
        return None

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluación del modelo")
    p.add_argument("--base_path", required=True, help="Ruta base del dataset con subcarpetas por clase")
    p.add_argument("--model_path", default=None, help="Ruta del modelo (SavedModel dir o .h5). Opcional.")
    p.add_argument("--img_size", default="224,224", help="Tamaño imagen WxH, ej: 224,224")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--validation_split", type=float, default=0.2)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--max_images_per_class", type=int, default=None)
    p.add_argument("--no_report", action="store_true", help="Desactiva classification report y matriz de confusión")
    p.add_argument("--save_report", default=None, help="Ruta a un JSON para guardar métricas/reportes")
    return p.parse_args()

def _to_tuple_hw(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--img_size debe tener formato W,H")
    return (parts[0], parts[1])

if __name__ == "__main__":
    import json as _json, os as _os
    args = parse_args()
    img_size = _to_tuple_hw(args.img_size)
    metrics, reports = evaluate_model(
        base_path=args.base_path,
        model_path=args.model_path,
        img_size=img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        test_split=args.test_split,
        max_images_per_class=args.max_images_per_class,
        classification_report_flag=not args.no_report,
        confusion_matrix_flag=not args.no_report,
    )

    print("Métricas:", metrics if metrics else "(vacío)")
    if reports:
        if "classification_report" in reports:
            print("\nClassification Report:\n")
            print(reports["classification_report"])
        if "confusion_matrix" in reports:
            print("\nConfusion Matrix:")
            for row in reports["confusion_matrix"]:
                print(row)
    else:
        print("No se generaron reportes adicionales.")

    if args.save_report:
        payload = {"metrics": metrics, "reports": reports}
        _os.makedirs(_os.path.dirname(args.save_report) or ".", exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados en: {args.save_report}")
