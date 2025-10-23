# Reporte de Optimización del Modelo

## 1. Contexto del Proyecto
El presente informe documenta el proceso de **optimización y análisis de sensibilidad** aplicado al modelo de diagnóstico temprano de **cáncer de tiroides mediante ecografía**.  
El estudio forma parte del proyecto “**Sistema de apoyo al diagnóstico basado en inteligencia artificial**”, desarrollado dentro de la *Maestría en Inteligencia Artificial* en la Universidad de Especialidades Espíritu Santo.

El objetivo principal fue mejorar el rendimiento del modelo base de clasificación mediante estrategias de búsqueda aleatoria de hiperparámetros (*RandomizedSearchCV*), evaluando su impacto en las métricas de validación y la interpretabilidad clínica.

---

## 2. Dataset y Preprocesamiento

**Tamaño total:** 637 imágenes ecográficas.  
**Etiquetas:** benignas y malignas.  
**Relación de clases:** 1.45:1 (moderadamente desbalanceado).  
**Formato:** imágenes preprocesadas (JPG/PNG) con extracción previa de *features* estadísticos y morfológicos.

### Variables analizadas
| Tipo | Variables | Descripción |
|------|------------|--------------|
| Morfológicas | Densidad de bordes, Asimetría | Describen contornos y forma del nódulo. |
| Radiométricas | Contraste, Intensidad promedio, Entropía, Curtosis | Características texturales y de brillo. |

El preprocesamiento incluyó:
- Normalización z-score por variable.  
- Eliminación de valores atípicos con *IQR filtering*.  
- Estandarización de features numéricos.  
- *Train-test split* estratificado (80/20) para evaluación consistente.  

---

## 3. Modelo Base y Metodología de Optimización

**Modelo inicial:** Random Forest Classifier (100 árboles, sin ajuste de pesos).  
**Objetivo de optimización:** maximizar *F1-score* ponderado en validación cruzada.  
**Método:** `RandomizedSearchCV` con validación estratificada (k=5).

### Espacio de hiperparámetros
| Parámetro | Rango explorado | Seleccionado |
|------------|-----------------|---------------|
| `n_estimators` | 100 – 1000 | 650 |
| `max_depth` | 5 – 50 | 22 |
| `min_samples_split` | 2 – 10 | 4 |
| `min_samples_leaf` | 1 – 5 | 2 |
| `max_features` | ['sqrt', 'log2'] | 'sqrt' |
| `class_weight` | ['balanced', None] | 'balanced' |

**Tiempo total de búsqueda:** 0.97 minutos.  
**Número de combinaciones evaluadas:** 50.

---

## 4. Análisis de Sensibilidad

Se evaluó el impacto individual de las variables mediante *tests de significancia (p-valor)*.

| Variable | p-valor | Interpretación |
|-----------|----------|----------------|
| Densidad de bordes | 5.88e-04 | Altamente significativo |
| Contraste | 3.91e-03 | Significativo |
| Intensidad promedio | 3.89e-02 | Moderadamente significativo |
| Asimetría | 7.82e-01 | No significativo |
| Entropía | 8.08e-01 | No significativo |
| Curtosis | 9.75e-01 | No significativo |

**Ranking de importancia (de mayor a menor):**  
1. Densidad de bordes  
2. Contraste  
3. Intensidad promedio  
4. Asimetría  
5. Entropía  
6. Curtosis  

Estas variables reflejan la relevancia de la **textura y los bordes irregulares** como factores determinantes en la detección de lesiones malignas.

---

## 5. Interacciones Críticas y Análisis Multivariante

Se identificó una **interacción complementaria** entre *contraste* y *densidad de bordes*, coherente con la diferenciación de lesiones malignas (bordes difusos y textura heterogénea).  
El análisis mediante **PCA (2 componentes)** evidenció una **separación parcial entre clases**, lo que sugiere que la combinación de variables mejora la capacidad discriminante aunque no de forma lineal perfecta.

---

## 6. Resultados y Comparación

| Métrica | Modelo Base | Modelo Optimizado | Mejora Relativa |
|----------|--------------|-------------------|------------------|
| Score Validación (F1) | 0.6051 | 0.6148 | +1.60% |
| Recall (maligno) | 0.70 | 0.73 | +4.3% |
| Precision (benigno) | 0.62 | 0.64 | +3.2% |
| Balanced Accuracy | 0.61 | 0.63 | +2.0% |

El modelo con **class weights balanceados** logró el mejor compromiso entre rendimiento y equidad, mejorando la sensibilidad hacia la clase minoritaria sin sacrificar precisión global.

---

## 7. Plan de Optimización Continua

| Fase | Periodo | Acciones Clave |
|------|----------|----------------|
| **Fase 1** | Semanas 1–2 | Revisión de outliers, rebalanceo con *oversampling*, validación cruzada estratificada. |
| **Fase 2** | Semanas 3–4 | Extensión de búsqueda a SVM, XGBoost y redes neuronales; calibración de probabilidades. |
| **Fase 3** | Semanas 5–6 | Validación clínica con radiólogos, análisis de sesgo (por equipo, paciente, centro). |

---

## 8. Visualizaciones Clave (descriptas)
- **Gráfico de importancia de variables (Feature Importance):** destaca densidad de bordes, contraste e intensidad media.  
- **Curva ROC y AUC:** mejora en área bajo la curva de 0.60 → 0.63.  
- **Matriz de confusión:** reducción de falsos negativos en la clase maligna.  
- **Proyección PCA 2D:** evidencia de agrupamientos parciales coherentes con las etiquetas.  

---

## 9. Conclusiones

- Se logró una **mejora del 1.6% en F1-score** y aumento del recall para nódulos malignos.  
- Las variables **densidad de bordes y contraste** constituyen predictores clínicamente relevantes.  
- El modelo optimizado es estadísticamente más estable y equitativo, aunque aún requiere:
  - Mayor tamaño de muestra.  
  - Inclusión de **CNNs o transfer learning** para extraer patrones espaciales complejos.  
  - Validación clínica regulada antes de uso diagnóstico real.

---

## 10. Próximos pasos recomendados

1. Implementar un modelo **CNN EfficientNet** con embeddings visuales supervisados.  
2. Aplicar **estrategias de data augmentation** en clases minoritarias.  
3. Desarrollar una **pipeline de calibración** de probabilidades (Platt / isotónica).  
4. Integrar Grad-CAM para interpretabilidad de las regiones activas.  
5. Registrar los experimentos y métricas en MLflow para trazabilidad.
