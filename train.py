"""
train.py - Script de entrenamiento

Extraído de Integrador31.ipynb y adaptado para ejecutarse como módulo standalone.
Incluye utilidades detectadas (p. ej., 'main_sistema_completo' si existe) y un
orquestador 'train_and_evaluate' con CLI que compone data_processing + model.
"""


# Utilidades de entrenamiento/evaluación extraídas del notebook

def main_sistema_completo():
    """Función principal del sistema completo - COMPLETAMENTE CORREGIDA"""
    print("🚀 INICIANDO SISTEMA COMPLETO DE ANÁLISIS TIROIDEO")
    print("=" * 60)
    print("🎯 MÓDULOS IMPLEMENTADOS:")
    print("   • EDA Avanzado con visualizaciones completas")
    print("   • Análisis de sesgo y matrices de confusión")
    print("   • Optimización de hiperparámetros con RandomizedSearchCV")
    print("   • Matriz de confusión balanceada destacada")
    print("   • Sistema de diagnóstico profesional con imágenes")

    # Inicializar componentes
    eda_analyzer = AdvancedEDA()
    bias_analyzer = BiasAnalysis()
    optimizer = HyperparameterOptimizer()

    try:
        # 1. Cargar datos
        print("\n📁 CARGANDO DATASET TIROIDEO...")
        X, y, df_metadatos = cargar_dataset_completo_avanzado()

        # Verificar que hay datos
        if len(X) == 0:
            print("❌ No se pudieron cargar datos. Usando dataset de ejemplo...")
            X, y, df_metadatos = crear_dataset_ejemplo()

        # 2. Análisis EDA completo
        print("\n🔍 EJECUTANDO ANÁLISIS EXPLORATORIO COMPLETO...")
        eda_results = eda_analyzer.perform_comprehensive_eda(X, y, df_metadatos)
        eda_report = eda_analyzer.generate_eda_report()

        # 3. Análisis de sesgo y matrices de confusión - CORREGIDO
        print("\n⚖️ REALIZANDO ANÁLISIS DE SESGO...")
        X_features = df_metadatos.select_dtypes(include=[np.number])

        # Seleccionar solo características relevantes que existan
        feature_cols = ['intensidad_promedio', 'contraste', 'entropia', 'asimetria',
                       'curtosis', 'densidad_bordes', 'magnitud_gradiente_promedio',
                       'hu_momento_1', 'hu_momento_2', 'heterogeneidad']

        # Filtrar características que realmente existen
        available_features = [col for col in feature_cols if col in X_features.columns]
        if not available_features:
            # Si no hay las características esperadas, usar las primeras numéricas
            available_features = X_features.select_dtypes(include=[np.number]).columns.tolist()[:5]

        X_filtered = X_features[available_features]

        # ✅ CORRECCIÓN APLICADA: Ahora perform_bias_analysis maneja todos los casos de error
        bias_results = bias_analyzer.perform_bias_analysis(X_filtered, y)

        if bias_results:
            bias_report = bias_analyzer.generate_bias_report()
        else:
            print("⚠️ No se pudo completar el análisis de sesgo")

        # 4. Optimización de hiperparámetros
        print("\n🎯 OPTIMIZANDO HIPERPARÁMETROS CON RANDOMIZEDSEARCHCV...")
        optimization_results, best_score = optimizer.perform_comprehensive_optimization(
            X_filtered, y, 'random_forest'
        )

        # 5. Reporte final de optimización
        optimization_report = optimizer.generate_optimization_report()

        print("\n🎉 SISTEMA COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        print("✅ TODOS LOS MÓDULOS EJECUTADOS:")
        print("   • EDA Avanzado con análisis estadístico")
        print("   • Matrices de confusión con múltiples técnicas de balanceo")
        print("   • Matriz de confusión balanceada destacada")
        print("   • Optimización sistemática con 30+ combinaciones")
        print("   • Sistema de diagnóstico profesional listo")

        return {
            'eda_results': eda_results,
            'bias_results': bias_results,
            'optimization_results': optimization_results,
            'dataset': (X, y, df_metadatos)
        }

    except Exception as e:
        print(f"❌ Error en el sistema completo: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_and_evaluate(
    base_path: str,
    img_size: tuple = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    max_images_per_class: int | None = None,
):
    """
    Orquesta el flujo de entrenamiento y evaluación usando data_processing y model.

    - Carga y preprocesa el dataset
    - Construye el modelo con `crear_modelo_prediccion_compatible`
    - Entrena el modelo
    - Evalúa en validación/test si está disponible
    Retorna (model, history, metrics_dict).
    """
    from data_processing import cargar_dataset_completo_avanzado
    from model import crear_modelo_prediccion_compatible

    # Carga de datos
    ds = cargar_dataset_completo_avanzado(
        base_path=base_path,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        test_split=test_split,
        max_images_per_class=max_images_per_class,
    )
    train_data = ds.get("train")
    val_data = ds.get("val")
    test_data = ds.get("test")

    # Crear modelo
    # Nota: ajustar num_classes según `CLASSES` si está disponible
    try:
        from data_processing import CLASSES
        num_classes = len(CLASSES)
    except Exception:
        # fallback: inferir de datos si es posible
        num_classes = getattr(train_data, "num_classes", None) or 2

    input_shape = (img_size[0], img_size[1], 3)
    model = crear_modelo_prediccion_compatible(input_shape=input_shape, num_classes=num_classes)

    # Entrenamiento (keras-like APIs)
    history = None
    if hasattr(model, "fit") and train_data is not None:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
        )

    # Evaluación
    metrics = {}
    evaluator = getattr(model, "evaluate", None)
    if callable(evaluator) and test_data is not None:
        try:
            results = model.evaluate(test_data, return_dict=True)
            if isinstance(results, dict):
                metrics.update(results)
        except TypeError:
            # versiones antiguas pueden no soportar return_dict
            res = model.evaluate(test_data)
            metrics["test_loss"] = res[0] if isinstance(res, (list, tuple)) and res else res

    return model, history, metrics


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Entrenamiento del modelo (train.py)")
    p.add_argument("--base_path", required=True, help="Ruta base del dataset con subcarpetas por clase")
    p.add_argument("--img_size", default="224,224", help="Tamaño de imagen WxH, ej: 224,224")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--validation_split", type=float, default=0.2)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--max_images_per_class", type=int, default=None)
    return p.parse_args()

def _to_tuple_hw(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--img_size debe tener formato W,H")
    return (parts[0], parts[1])

if __name__ == "__main__":
    args = parse_args()
    img_size = _to_tuple_hw(args.img_size)
    model, history, metrics = train_and_evaluate(
        base_path=args.base_path,
        img_size=img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        test_split=args.test_split,
        max_images_per_class=args.max_images_per_class,
    )

    # Guardado opcional del modelo si la API lo soporta
    try:
        model.save("model_saved")
        print("Modelo guardado en ./model_saved")
    except Exception as e:
        print(f"No se pudo guardar el modelo automáticamente: {e}")

    if metrics:
        print("Métricas de test:", metrics)
    else:
        print("Entrenamiento finalizado. No se reportaron métricas de test.")
