import streamlit as st
from utils_logs import LogCapture
# Importa tus funciones desde tu archivo real
from integrador33 import ejecutar_analisis_eda

st.set_page_config(page_title="EDA Tiroideo", page_icon="🔍")
st.title("🔍 Análisis EDA")

st.write("Pulsa para ejecutar el EDA. Se mostrarán gráficos y logs capturados.")
if st.button("Ejecutar EDA"):
    with LogCapture() as cap:
        ejecutar_analisis_eda()  # Esta función ya hace st.pyplot(...) internamente
    logs = cap.text()
    with st.expander("📜 Ver logs de consola (EDA)"):
        st.code(logs, language="bash")
    st.download_button("⬇️ Descargar logs EDA", logs, file_name="logs_eda.txt")
