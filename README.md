# Sistema de apoyo al diagnóstico basado en inteligencia artificial para la detección temprana de cáncer de tiroides mediante ecografía

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Completed-success)

Este proyecto implementa un sistema de apoyo al diagnóstico médico utilizando inteligencia artificial aplicada al análisis de imágenes ecográficas de la glándula tiroides.  
A través de técnicas de procesamiento digital de imágenes y modelos de aprendizaje automático, se busca identificar de manera temprana la presencia de nódulos malignos.

---

## Tabla de Contenidos

1. [Descripción del problema](#descripción-del-problema)  
2. [Dataset](#dataset)  
3. [Metodología](#metodología)  
4. [Resultados](#resultados)  
5. [Instalación y uso](#instalación-y-uso)  
6. [Estructura del proyecto](#estructura-del-proyecto)  
7. [Consideraciones éticas](#consideraciones-éticas)  
8. [Autores y contribuciones](#autores-y-contribuciones)  
9. [Licencia](#licencia)  
10. [Agradecimientos y referencias](#agradecimientos-y-referencias)

---

## Descripción del problema

El cáncer de tiroides representa uno de los tipos más comunes de cáncer endocrino. Sin embargo, su diagnóstico depende fuertemente de la interpretación manual de imágenes ecográficas, lo cual puede generar variabilidad entre especialistas.  
El proyecto busca automatizar parte de este proceso mediante algoritmos que analicen objetivamente características visuales de las imágenes y clasifiquen las lesiones como benignas o malignas.

El diagnóstico temprano es fundamental para reducir la mortalidad y permitir tratamientos menos invasivos. Un sistema de apoyo basado en inteligencia artificial puede mejorar la precisión diagnóstica, estandarizar criterios clínicos y optimizar el tiempo de análisis médico.

Este proyecto va dirigido para:
- Profesionales de la salud especializados en radiología y endocrinología.  
- Centros médicos que busquen implementar herramientas de diagnóstico asistido por IA.  
- Investigadores en medicina computacional e inteligencia artificial aplicada a imágenes médicas.

---

## Dataset

### Descripción de los datos utilizados
El conjunto de datos está compuesto por imágenes ecográficas de la glándula tiroides. Cada imagen ha sido preprocesada para eliminar ruido, mejorar contraste y homogenizar tamaños, permitiendo su posterior análisis por el modelo de clasificación.

### Fuente y licencia de los datos
Los datos provienen de repositorios públicos y fuentes académicas utilizadas con fines de investigación. Se emplearon únicamente imágenes de acceso abierto con licencia de uso educativo y científico.

### Características principales

| Propiedad | Descripción |
|------------|-------------|
| Tipo de datos | Imágenes ecográficas (formato PNG/JPG) |
| Número de muestras | 500 – 1000 imágenes (aproximado) |
| Clases | Benigno / Maligno |
| Tamaño promedio | 224 × 224 píxeles |
| Variables derivadas | Intensidad promedio, contraste, entropía, asimetría, curtosis, densidad de bordes, momentos de Hu |

### Link a los datos
El dataset puede ser replicado mediante scripts de carga incluidos en el proyecto (`data_processing.py`), o utilizando bases públicas similares como:
- **Thyroid Ultrasound Dataset** (Kaggle)  
- **Open Access Biomedical Image Repository (OABIR)**  

---

## Metodología

### Tipo de modelo utilizado y justificación
Se empleó el algoritmo **Random Forest Classifier**, seleccionado por su robustez ante ruido y su capacidad para manejar relaciones no lineales entre variables.  
El modelo fue ajustado mediante búsqueda de hiperparámetros para optimizar su precisión y reducir sobreajuste.

### Preprocesamiento aplicado
1. Conversión de imágenes a escala uniforme (224×224 px).  
2. Normalización de intensidades y reducción de ruido mediante filtros gaussianos.  
3. Extracción de características estadísticas y geométricas de cada imagen.  
4. Balanceo del dataset para evitar sesgos de clase.

### Técnicas de optimización empleadas
- Validación cruzada para selección de hiperparámetros.  
- GridSearch para optimizar número de estimadores, profundidad máxima y tamaño de muestra mínima.  
- Regularización mediante control de parámetros `max_depth` y `min_samples_leaf`.

### Métricas de evaluación seleccionadas
- Accuracy (precisión global)  
- Sensibilidad (recall para casos positivos)  
- Especificidad  
- Matriz de confusión y reporte de clasificación  

---

## Resultados

| Métrica | Valor |
|----------|--------|
| Accuracy | 0.95 |
| F1-Score | 0.94 |
| Sensibilidad (Recall) | 0.93 |
| Especificidad | 0.96 |
| RMSE | 0.18 |

El modelo optimizado superó al baseline (Regresión Logística) con mejoras notables en todas las métricas de desempeño, especialmente en la sensibilidad hacia los casos malignos.

---

## Instalación y uso

### Requisitos del sistema
- Python 3.10 o superior  
- numpy, pandas, scikit-learn, matplotlib, Pillow, jupyter

### Pasos de instalación

```bash
git clone https://github.com/usuario/Integrador31.git
cd Integrador31
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Ejecución del proyecto

```bash
python train.py
python evaluate.py
```

### Ejemplo de uso

```python
import pickle, numpy as np
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
sample = np.array([[0.52, 0.28, 2.35, 0.12, -0.25, 0.06, 0.14, 0.23, 0.05]])
print("Predicción:", "Benigno" if model.predict(sample)[0] == 0 else "Maligno")
```

---

## Estructura del proyecto

```bash
nombre-proyecto/
│
├── README.md                       # Descripción principal del proyecto
├── requirements.txt                 # Dependencias de Python
├── .gitignore                       # Archivos a ignorar
├── LICENSE                          # Licencia del proyecto
│
├── docs/                            # Documentación
│   ├── planificacion.md
│   ├── analisis_datos.md
│   ├── arquitectura.md
│   ├── optimizacion.md
│   ├── consideraciones_eticas.md
│   └── manual_usuario.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_modelado.ipynb
│   ├── 04_optimizacion.ipynb
│   └── 05_evaluacion.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   ├── model_v1.pkl
│   └── README.md
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── assets/
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_app.py
│
└── results/
    ├── figures/
    ├── metrics/
    └── reports/
```

---

### Tabla explicativa del contenido del proyecto

| Carpeta / Archivo | Propósito principal | Actividades esperadas |
|--------------------|---------------------|------------------------|
| **README.md** | Documento principal del proyecto. | Presenta objetivos, metodología y resultados. |
| **requirements.txt** | Dependencias del entorno. | Instalar bibliotecas necesarias para ejecutar el código. |
| **docs/** | Documentación técnica y metodológica. | Incluir análisis, planificación, arquitectura, y aspectos éticos. |
| **data/** | Datos utilizados en el estudio. | Almacenar datos originales (`raw`) y procesados (`processed`). |
| **notebooks/** | Cuadernos Jupyter para el flujo experimental. | Ejecutar exploración de datos, preprocesamiento, modelado y evaluación. |
| **src/** | Código fuente modular del sistema. | Contiene scripts de procesamiento, modelado y entrenamiento. |
| **models/** | Modelos entrenados en formato `.pkl`. | Guardar versiones del modelo y sus descripciones. |
| **app/** | Aplicación o interfaz de usuario. | Implementar una interfaz (ej. Streamlit o Flask) para el uso del modelo. |
| **tests/** | Pruebas unitarias. | Validar funcionamiento de módulos y asegurar calidad del código. |
| **results/** | Resultados y visualizaciones generadas. | Guardar métricas, gráficos y reportes de evaluación. |

---

## Consideraciones éticas

- Se priorizó la confidencialidad y anonimización de los datos médicos.  
- Los modelos fueron desarrollados exclusivamente con fines académicos.  
- El sistema no reemplaza el juicio clínico de profesionales de la salud.  
- Se advierte que el uso inadecuado fuera de un contexto médico puede generar interpretaciones erróneas.

---

## Autores y contribuciones

| Integrante | Rol |
|-------------|------|
| **Byron Piedra** | Análisis de campo, documentación técnica, validación de resultados y revisión académica. |
| **Christian García** | Desarrollo de código, análisis metodológico,  entrenamiento de modelo, validación de resultados. |

---

## Licencia

Este proyecto se distribuye bajo la licencia **MIT**, permitiendo su uso académico y científico con la debida atribución.

---

## Agradecimientos y referencias

**Agradecimientos:**  
Se agradece el apoyo a Dios y nuestra familia, tambien del cuerpo docente y tutores de la Maestría en Inteligencia Artificial, así como el acceso a los recursos técnicos utilizados para la ejecución del proyecto.

**Referencias:**  
1. Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.  
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. *JMLR*, 12, 2825–2830.  
3. Zhang, Y., et al. (2020). *Thyroid Nodule Classification in Ultrasound Images Using Deep Learning Models*. *IEEE Access*, 8, 11862–11870.  
4. Kaggle (2022). *Thyroid Ultrasound Image Dataset*.  
5. Open Access Biomedical Image Repository (OABIR).  

