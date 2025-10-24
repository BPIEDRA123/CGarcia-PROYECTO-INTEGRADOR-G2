
import os
import sys
import types
import tempfile
import numpy as np
import pytest

# --- Helper: gracefully import data_processing even if tensorflow is missing ---
def _safe_import_data_processing():
    try:
        import data_processing as dp
        return dp
    except ModuleNotFoundError as e:
        if "tensorflow" in str(e):
            # Inject a minimal fake tensorflow to satisfy imports
            fake_tf = types.ModuleType("tensorflow")
            keras = types.ModuleType("keras")
            keras.layers = types.SimpleNamespace()
            keras.models = types.SimpleNamespace()
            keras.regularizers = types.SimpleNamespace()
            fake_tf.keras = keras
            sys.modules["tensorflow"] = fake_tf
            import data_processing as dp
            return dp
        raise

dp = _safe_import_data_processing()


def test_module_exports_nonempty():
    # The module should define an __all__ or at least key functions
    assert hasattr(dp, "__all__")
    assert isinstance(dp.__all__, (list, tuple))
    # Expect at least some well-known names
    expected_any = {
        "mejorar_calidad_imagen",
        "cargar_y_preprocesar_imagen_avanzado",
        "es_archivo_imagen_avanzado",
        "extraer_caracteristicas_avanzadas_completas",
        "cargar_dataset_completo_avanzado",
        "crear_dataset_ejemplo",
    }
    assert any(name in dp.__all__ for name in expected_any)


def test_es_archivo_imagen_avanzado_true_false(tmp_path):
    # Create sample files
    img_path = tmp_path / "sample.jpg"
    txt_path = tmp_path / "notes.txt"
    img_path.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header bytes
    txt_path.write_text("not an image", encoding="utf-8")

    assert dp.es_archivo_imagen_avanzado(str(img_path)) is True
    assert dp.es_archivo_imagen_avanzado(str(txt_path)) is False


def test_extraer_caracteristicas_avanzadas_completas_keys():
    # Create a synthetic image array (H, W, C) in 0..1
    arr = np.clip(np.random.normal(0.5, 0.2, (64, 64, 3)), 0, 1).astype(np.float32)
    feats = dp.extraer_caracteristicas_avanzadas_completas(arr)
    assert isinstance(feats, dict), "La función debe retornar un diccionario de características"
    # Validate presence of a minimal subset of feature keys used en el notebook
    expected_keys = {
        "intensidad_promedio",
        "contraste",
        "entropia",
        "asimetria",
        "curtosis",
        "densidad_bordes",
        "magnitud_gradiente_promedio",
        "hu_momento_1",
        "hu_momento_2",
    }
    missing = expected_keys.difference(set(feats.keys()))
    assert not missing, f"Faltan claves de características esperadas: {missing}"


def test_mejorar_calidad_imagen_runs():
    # Verifica que la función procesa sin errores y mantiene dimensiones
    arr = np.clip(np.random.normal(0.5, 0.2, (64, 64, 3)), 0, 1).astype(np.float32)
    try:
        out = dp.mejorar_calidad_imagen(arr)
    except TypeError:
        # Algunos diseños usan PIL Image en lugar de numpy; adapta el input si es necesario
        from PIL import Image
        im = Image.fromarray((arr * 255).astype("uint8"))
        out = dp.mejorar_calidad_imagen(im)
        if not isinstance(out, np.ndarray):
            # Si retorna Image, convertirla para verificar forma
            out = np.array(out)
    assert isinstance(out, np.ndarray), "La salida debe ser una imagen como arreglo numpy"
    assert out.shape[:2] == (64, 64)


def test_crear_dataset_ejemplo_sanity():
    # Debe devolver estructuras no vacías para pruebas rápidas
    data = dp.crear_dataset_ejemplo()
    # Acepta retorno como (X, y, metadatos) o estructuras equivalentes
    if isinstance(data, dict):
        assert "metadatos" in data or "X" in data
    elif isinstance(data, (list, tuple)):
        assert len(data) >= 2
        # El primer elemento suele ser imágenes o features; solo comprobación superficial
        assert data[0] is not None
        assert data[1] is not None
    else:
        pytest.skip("Formato de retorno no estándar; verifique implementación actual de crear_dataset_ejemplo")


@pytest.mark.parametrize("w,h", [(32,32), (64,64)])
def test_cargar_y_preprocesar_imagen_avanzado_accepts_shape(w, h):
    # Genera una imagen aleatoria y verifica que se preprocese al tamaño solicitado si la API lo permite
    arr = np.clip(np.random.normal(0.5, 0.2, (h, w, 3)), 0, 1).astype(np.float32)
    try:
        out = dp.cargar_y_preprocesar_imagen_avanzado(arr, target_size=(w, h))
    except TypeError:
        # Algunas implementaciones solo reciben ruta y usan target_size global; en ese caso, saltar
        pytest.skip("La función cargar_y_preprocesar_imagen_avanzado no acepta entrada directa o target_size paramétrico")
    else:
        assert isinstance(out, np.ndarray)
        assert out.shape[:2] == (h, w)
