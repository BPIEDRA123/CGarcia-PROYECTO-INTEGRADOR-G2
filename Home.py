import streamlit as st

st.set_page_config(page_title="Panel Tiroideo con IA", page_icon="🧠")

st.title("Panel de Análisis Tiroideo con IA 🧠")
st.write("Elige qué módulo abrir en una pestaña aparte:")

# Opción A: enlaces nativos (si tu versión de Streamlit lo soporta)
try:
    # Desde Streamlit 1.32 existe st.page_link; en algunas versiones incluye new_tab
    st.page_link("pages/1_🔍_EDA.py", label="🔍 Abrir EDA en nueva pestaña", icon="🔎", new_tab=True)  # si tu versión lo soporta
    st.page_link("pages/2_🧠_Entrenamiento.py", label="🧠 Abrir Entrenamiento en nueva pestaña", icon="🤖", new_tab=True)
    st.page_link("pages/3_📷_Predicción_por_Imagen.py",
                 label="📷 Abrir Predicción por Imagen en nueva pestaña",
                 icon="📷", new_tab=True)
except TypeError:
    # Opción B: HTML seguro como fallback (abre SIEMPRE en pestaña nueva)
    st.markdown(
        '<a href="pages/1_%F0%9F%94%8D_EDA" target="_blank">🔍 Abrir EDA en nueva pestaña</a>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<a href="pages/2_%F0%9F%A7%A0_Entrenamiento" target="_blank">🧠 Abrir Entrenamiento en nueva pestaña</a>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<a href="pages/3_%F0%9F%93%B7_Predicci%C3%B3n_por_Imagen" target="_blank">'
        '📷 Abrir Predicción por Imagen en nueva pestaña</a>',
        unsafe_allow_html=True
    )

st.info("Tip: también puedes **Ctrl/Cmd + clic** sobre el enlace para abrir en nueva pestaña.")
