# pages/Entrenamiento.py
import streamlit as st
from utils_logs import LogCapture
from integrador33 import main_sistema_completo

st.set_page_config(page_title="Entrenamiento", page_icon="🧠")
st.title("🧠 Entrenamiento de Modelos")

st.write("Pulsa para ejecutar el entrenamiento. Se mostrarán gráficos y logs.")
if st.button("Ejecutar Entrenamiento"):
    with LogCapture() as cap:
        resultado = main_sistema_completo()   # ✅ devuelve un diccionario

    logs = cap.text()
    with st.expander("📜 Ver logs de consola (Entrenamiento)"):
        st.code(logs, language="bash")
    st.download_button("⬇️ Descargar logs", logs, file_name="logs_entrenamiento.txt")

    if not resultado:
        st.error("❌ Ocurrió un error en el entrenamiento.")
    else:
        # resultado es un dict con llaves: 'eda_results', 'bias_results', 'optimization_results', 'dataset'
        st.success("✅ Entrenamiento ejecutado.")

        eda_ok = resultado.get('eda_results') is not None
        bias_ok = resultado.get('bias_results') is not None
        opt_ok = resultado.get('optimization_results') is not None

        st.write(f"- **EDA**: {'OK' if eda_ok else 'No disponible'}")
        st.write(f"- **Bias**: {'OK' if bias_ok else 'No disponible'}")
        st.write(f"- **Optimización**: {'OK' if opt_ok else 'No disponible'}")

        # Muestra info básica del dataset devuelto
        X, y, df_metadatos = resultado.get('dataset', (None, None, None))
        if X is not None and y is not None and df_metadatos is not None:
            st.write(f"- **Tamaño X**: {getattr(X, 'shape', None)}")
            st.write(f"- **Tamaño y**: {len(y)}")
            st.write(f"- **df_metadatos**: {df_metadatos.shape}")
