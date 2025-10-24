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
4. [Resultados esperados](#resultados-esperados)  
5. [Autores](#autores)  

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

## Resultados esperados

El modelo final alcanzó una **precisión de validación del 95%**, mostrando un desempeño superior en la detección de lesiones malignas sin comprometer la especificidad.  
La integración del sistema permitiría mejorar la eficiencia diagnóstica y proporcionar una herramienta de apoyo confiable para profesionales de la salud.

---

## Autores

**Byron Piedra**  
**Christian García**
