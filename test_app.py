# -*- coding: utf-8 -*-
"""
test_app.py - Pruebas de las páginas Streamlit (EDA y Entrenamiento) y utilidades de logs.
=========================================================================================

Ejecuta con:
    pytest -q test_app.py
"""

import sys
import types
import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------
# Fakes de Streamlit (API mínima necesaria)
# ---------------------------------------------------------------------

class DummyExpander:
    def __init__(self, app):
        self.app = app
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def code(self, text, language=None):
        self.app._calls.append(("code", text, language))
        self.app.last_code = (text, language)

class FakeStreamlit:
    def __init__(self):
        self._calls = []
        self._button_next = False
        self.last_title = None
        self.last_writes = []
        self.last_success = None
        self.last_error = None
        self.last_download = None
        self.last_code = None

    def set_button_return(self, value: bool):
        self._button_next = value

    def set_page_config(self, **kwargs):
        self._calls.append(("set_page_config", kwargs))

    def title(self, text):
        self.last_title = text
        self._calls.append(("title", text))

    def write(self, text):
        self.last_writes.append(text)
        self._calls.append(("write", text))

    def button(self, label):
        self._calls.append(("button", label))
        return self._button_next

    def expander(self, label):
        self._calls.append(("expander", label))
        return DummyExpander(self)

    def download_button(self, label, data, file_name=None):
        self.last_download = (label, data, file_name)
        self._calls.append(("download_button", label, file_name))

    def success(self, text):
        self.last_success = text
        self._calls.append(("success", text))

    def error(self, text):
        self.last_error = text
        self._calls.append(("error", text))

# ---------------------------------------------------------------------
# Utilidad: módulo backend simulado 'integrador33'
# ---------------------------------------------------------------------

class DummyIntegrador33(types.ModuleType):
    def __init__(self):
        super().__init__("integrador33")
        self._eda_called = False
        self._train_called = False

    def ejecutar_analisis_eda(self):
        print("EDA: inicio")
        print("EDA: fin")
        self._eda_called = True

    def main_sistema_completo(self):
        print("TRAIN: inicio")
        print("TRAIN: fin")
        self._train_called = True
        import numpy as np
        import pandas as pd
        X = np.zeros((10, 5))
        y = [0]*5 + [1]*5
        df = pd.DataFrame({"a":[1,2,3]})
        return {
            "eda_results": {"ok": True},
            "bias_results": {"ok": True},
            "optimization_results": {"ok": True},
            "dataset": (X, y, df),
        }

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_utils_logs_basic_capture():
    try:
        import utils_logs as UL
    except Exception:
        pytest.skip("utils_logs.py no disponible.")
        return

    with UL.LogCapture() as cap:
        print("hola")
        print("mundo")
    text = cap.text()
    assert "hola" in text and "mundo" in text

def _import_by_candidate_paths(module_name, paths):
    import importlib.util
    for p in paths:
        if Path(p).exists():
            spec = importlib.util.spec_from_file_location(module_name, str(p))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            return mod
    return None

def test_eda_page_calls_backend(monkeypatch):
    fake_st = FakeStreamlit()
    dummy_backend = DummyIntegrador33()

    # Inyecta streamlit fake
    st_mod = types.ModuleType("streamlit")
    st_mod.set_page_config = fake_st.set_page_config
    st_mod.title = fake_st.title
    st_mod.write = fake_st.write
    st_mod.button = fake_st.button
    st_mod.expander = fake_st.expander
    st_mod.download_button = fake_st.download_button
    sys.modules["streamlit"] = st_mod

    # Inyecta backend dummy
    sys.modules["integrador33"] = dummy_backend

    fake_st.set_button_return(True)

    mod = _import_by_candidate_paths(
        "EDA",
        ["EDA.py", "pages/EDA.py"]
    )
    if mod is None:
        pytest.skip("EDA.py no encontrado.")
        return

    assert dummy_backend._eda_called is True
    assert fake_st.last_title is not None
    assert fake_st.last_download is not None
    assert fake_st.last_code is not None

def test_entrenamiento_page_happy_path(monkeypatch):
    fake_st = FakeStreamlit()
    dummy_backend = DummyIntegrador33()

    st_mod = types.ModuleType("streamlit")
    st_mod.set_page_config = fake_st.set_page_config
    st_mod.title = fake_st.title
    st_mod.write = fake_st.write
    st_mod.button = fake_st.button
    st_mod.expander = fake_st.expander
    st_mod.download_button = fake_st.download_button
    st_mod.success = fake_st.success
    st_mod.error = fake_st.error
    sys.modules["streamlit"] = st_mod

    sys.modules["integrador33"] = dummy_backend
    fake_st.set_button_return(True)

    mod = _import_by_candidate_paths(
        "Entrenamiento",
        ["Entrenamiento.py", "pages/Entrenamiento.py"]
    )
    if mod is None:
        pytest.skip("Entrenamiento.py no encontrado.")
        return

    assert dummy_backend._train_called is True
    assert fake_st.last_success is not None
    joined_writes = "\n".join([str(w) for w in fake_st.last_writes])
    assert "EDA" in joined_writes and "Bias" in joined_writes and "Optimización" in joined_writes
    assert fake_st.last_download is not None
