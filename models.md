# Modelos de Clasificación - Proyecto Integrador31

Este documento describe las versiones de los modelos entrenados y serializados en formato `.pkl` como parte del proyecto **Sistema de apoyo al diagnóstico basado en inteligencia artificial para la detección temprana de cáncer de tiroides mediante ecografía**.

---

## Descripción General

Los modelos fueron entrenados siguiendo el flujo establecido en el cuaderno `Integrador31.ipynb`, empleando el algoritmo **Random Forest Classifier** de la biblioteca *Scikit-learn*.  
El entrenamiento se realizó sobre características extraídas de imágenes ecográficas procesadas, con el objetivo de desarrollar un sistema de apoyo al diagnóstico que permita la clasificación temprana de posibles patologías tiroideas.

Cada versión del modelo representa una etapa del proceso de desarrollo y optimización, con ajustes en los hiperparámetros y en la estructura de los datos de entrada.

---

## Modelos Incluidos

| Archivo | Versión | Descripción | Accuracy de validación | Observaciones |
|----------|----------|--------------|------------------------|----------------|
| `model_v1.pkl` | v1 | Versión inicial del modelo antes del proceso de optimización. | ~0.90 | Entrenamiento base con configuración predeterminada. |
| `best_model.pkl` | v2 (actual) | Modelo optimizado con parámetros ajustados y mejor rendimiento. | ~0.95–0.97 | Versión recomendada para inferencia y evaluación. |

---

## Estructura Interna de los Archivos `.pkl`

Cada archivo `.pkl` contiene un diccionario serializado con la siguiente estructura:

```python
{
  "model": RandomForestClassifier(...),
  "feature_names": [
    "intensidad_promedio", "contraste", "entropia", "asimetria",
    "curtosis", "densidad_bordes", "magnitud_gradiente_promedio",
    "hu_momento_1", "hu_momento_2"
  ],
  "validation_accuracy": 0.95,
  "version": "v2",
  "description": "Versión optimizada del modelo."
}
```

---

## Instrucciones de Uso

### Cargar un modelo
```python
import pickle

with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
features = data["feature_names"]
print("Accuracy de validación:", data["validation_accuracy"])
```

### Realizar una predicción
```python
import numpy as np

# Ejemplo de entrada (mismo orden de las características)
sample = np.array([[0.52, 0.28, 2.35, 0.12, -0.25, 0.06, 0.14, 0.23, 0.05]])
pred = model.predict(sample)
print("Predicción:", "Benigno" if pred[0] == 0 else "Maligno")
```

---

## Requisitos del Entorno

- Python 3.10 o superior  
- Scikit-learn ≥ 1.3  
- Numpy, Pandas  

---

## Versionado

- **v1:** Modelo base inicial entrenado con parámetros por defecto.  
- **v2 (best_model):** Modelo optimizado con ajuste de hiperparámetros y mejor desempeño predictivo.  

---

## Autores

**Byron Piedra**  
**Christian García**  

---

Fecha de actualización: Octubre 2025
