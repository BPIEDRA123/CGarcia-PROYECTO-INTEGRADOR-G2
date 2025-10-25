# pages/3_📷_Predicción_por_Imagen.py
import streamlit as st
import io, contextlib

from integrador33 import sistema_prediccion_doctor

st.set_page_config(page_title="Predicción por Imagen", page_icon="📷")
st.title("📷 Predicción por Imagen")
st.write("Sube una imagen tiroidea y ejecuta el diagnóstico asistido por IA.")

# Captura de logs de todo lo que imprima el método
log_buffer = io.StringIO()
with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
    sistema_prediccion_doctor()  # ← sin botón

logs = log_buffer.getvalue()
with st.expander("📜 Logs del módulo"):
    st.code(logs or "Sin salida.", language="bash")

st.download_button(
    "⬇️ Descargar logs",
    data=logs or "Sin salida.",
    file_name="logs_prediccion.txt"
)
