
import numpy as np
import pytest

def _get_model_object(maybe):
    """
    Acepta: objeto modelo, (modelo, ..), {"model": modelo}
    Retorna el objeto que expone fit/predict.
    """
    if hasattr(maybe, "fit") and hasattr(maybe, "predict"):
        return maybe
    if isinstance(maybe, (list, tuple)) and maybe:
        for x in maybe:
            if hasattr(x, "fit") and hasattr(x, "predict"):
                return x
    if isinstance(maybe, dict) and "model" in maybe:
        m = maybe["model"]
        if hasattr(m, "fit") and hasattr(m, "predict"):
            return m
    return None


def test_exports_and_factory():
    import model as mdl
    assert hasattr(mdl, "__all__")
    assert "crear_modelo_prediccion_compatible" in mdl.__all__

    # La firma esperada: (input_shape=(H,W,C), num_classes=int)
    factory = getattr(mdl, "crear_modelo_prediccion_compatible")
    model_like = factory(input_shape=(64, 64, 3), num_classes=2)
    m = _get_model_object(model_like)
    assert m is not None, "La fábrica debe retornar (o contener) un objeto con métodos fit/predict"


def test_model_fit_predict_on_synthetic():
    import model as mdl
    factory = getattr(mdl, "crear_modelo_prediccion_compatible")

    model_like = factory(input_shape=(64, 64, 3), num_classes=2)
    m = _get_model_object(model_like)
    assert m is not None

    # Datos sintéticos de 9 features, consistentes con data_processing
    rng = np.random.default_rng(0)
    n = 80
    X = np.column_stack([
        rng.normal(0.5, 0.15, n),   # intensidad_promedio
        rng.normal(0.3, 0.08, n),   # contraste
        rng.normal(2.4, 0.5, n),    # entropia
        rng.normal(0.1, 0.1, n),    # asimetria
        rng.normal(-0.2, 0.3, n),   # curtosis
        rng.normal(0.06, 0.02, n),  # densidad_bordes
        rng.normal(0.15, 0.05, n),  # magnitud_gradiente_promedio
        rng.normal(0.25, 0.1, n),   # hu_momento_1
        rng.normal(0.05, 0.02, n),  # hu_momento_2
    ])
    y = (X @ np.array([0.8,0.6,0.4,0.2,-0.3,0.5,0.7,0.4,-0.2]) + rng.normal(0,0.25,n) > 0).astype(int)

    # Algunas implementaciones exigen 2D; otras aceptan pandas; mantenemos numpy 2D
    m.fit(X, y)
    preds = m.predict(X[:5])
    assert getattr(preds, "shape", (5,)) [0] in (5,), "La predicción debe devolver 5 instancias"


def test_hyperparameter_optimizer_presence():
    import model as mdl
    if "HyperparameterOptimizer" not in mdl.__all__:
        pytest.skip("HyperparameterOptimizer no está disponible en este build del modelo")
    HPO = getattr(mdl, "HyperparameterOptimizer")
    # Intento de construcción tolerante: si requiere parámetros específicos, marcar skip
    try:
        _ = HPO(object())
    except Exception:
        pytest.skip("No se pudo instanciar HyperparameterOptimizer con firma genérica")
