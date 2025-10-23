
# Planificación del Proyecto Integrador

## Plan de Acción (4 Semanas)

### Semana 1: Preparación de Datos y Pipeline Base
- Auditoría del dataset recomendado (*Thyroid_Ultrasound_2*)
- Implementación de técnicas básicas de aumento de datos (data augmentation)
- División de los datos en *train/validation/test* (70/20/10)
- Configuración del entorno de desarrollo (Colab, GitHub, dependencias)
- **Entregables:** Dataset balanceado, repositorio inicial, pipeline reproducible

### Semana 2: Entrenamiento Baseline
- Entrenamiento del modelo base con *EfficientNetB0* o *ResNet50*
- Implementación de aumentos adicionales de datos
- Configuración de notebook o CLI para inferencia básica
- Pruebas unitarias del *dataloader*
- **Entregables:** Modelo baseline entrenado, notebook de inferencia, pruebas unitarias

### Semana 3: Optimización y Validación
- Ajuste de hiperparámetros (learning rate, batch size, freezing)
- Implementación de *Grad-CAM* para interpretabilidad
- Validación del modelo con matriz de confusión
- Análisis de errores y documentación de resultados
- **Entregables:** Modelo optimizado, mapas Grad-CAM, reporte de validación

### Semana 4: Documentación y Entrega Final
- Exportación del modelo a formato *ONNX/TFLite* para CPU estándar
- Pruebas integrales (unitarias, integración, regresión)
- Elaboración de documentación técnica y manual de usuario
- Preparación de la presentación y demostración final
- **Entregables:** Modelo final exportado, manuales técnicos, presentación final

---

## Cronograma con Metodología Ágil

| Sprint | Semana | Objetivo Principal                | Entregables Clave |
|:------:|:-------:|----------------------------------|-------------------|
| 1 | Semana 1 | Preparación de datos y pipeline base | Dataset balanceado, pipeline reproducible |
| 2 | Semana 2 | Entrenamiento baseline | Modelo base entrenado, pruebas unitarias |
| 3 | Semana 3 | Optimización y validación | Modelo optimizado, Grad-CAM, métricas finales |
| 4 | Semana 4 | Documentación y entrega final | Modelo exportado, documentación, demo funcional |

---

## Plan de Recursos

### Recursos Humanos

| Rol | Responsabilidades | Horas Estimadas | Disponibilidad |
|------|------------------|-----------------|----------------|
| Desarrollador ML | Preprocesamiento, entrenamiento y optimización del modelo | 120 h | Total |
| Analista de Datos | Auditoría, métricas y visualizaciones | 80 h | Parcial |
| Project Manager | Coordinación y comunicación con stakeholders | 40 h | Total |
| Radiólogo Consultor | Validación clínica y retroalimentación | 10 h | Externo |
| Especialista Cloud | Configuración de entorno y soporte técnico | 5 h | Externo |

### Recursos Técnicos

| Recurso | Especificaciones | Proveedor | Costo |
|----------|------------------|------------|-------|
| GPU Cloud | NVIDIA V100, 16GB VRAM | Google Colab Pro | 50 USD/semana |
| Almacenamiento | 100 GB SSD | Google Drive | Incluido |
| Backup Local | 500 GB HDD | Interno | 0 USD |

### Recursos Financieros

| Categoría | Costo Estimado (USD) | Justificación |
|------------|----------------------|----------------|
| Consultoría Externa | 700 | Radiólogo y especialista cloud |
| Infraestructura Cloud | 100 | Subscripción Colab Pro y almacenamiento |
| Imprevistos (15%) | 120 | Contingencias varias |
| **Total Estimado** | **920 USD** | — |

---

## Hitos y Entregables

| Hito | Semana | Entregable | Criterio de Aceptación | Responsable |
|------|---------|-------------|------------------------|--------------|
| 1 | 1 | Auditoría del dataset | Informe de calidad y balanceo | Desarrollador |
| 2 | 1 | Pipeline reproducible | División 70/20/10 validada | Desarrollador / PO |
| 3 | 2 | Modelo baseline | F1 ≥ 0.82 en validación | Desarrollador |
| 4 | 2 | Inferencia inicial | < 5 s por imagen en CPU | Desarrollador |
| 5 | 3 | Optimización de hiperparámetros | F1 ≥ 0.85, Recall ≥ 0.90 | Desarrollador |
| 6 | 3 | Implementación de Grad-CAM | Visualización interpretable | Dev / Consultor clínico |
| 7 | 4 | Exportación final y pruebas | Modelo ejecuta en laptop estándar | Desarrollador |
| 8 | 4 | Documentación y presentación | Manual técnico y demo aprobada | PO / Dev |

---

## Métricas de Éxito

### Métricas Técnicas

| Métrica | Mínimo Aceptable | Objetivo de Excelencia | Benchmark |
|----------|------------------|------------------------|------------|
| F1-Score (macro) | 0.85 | 0.92 | 0.88–0.90 |
| Recall (maligno) | 0.90 | 0.95 | 0.85 (humano) |
| Precision (benigno) | 0.85 | 0.90 | 0.80 (humano) |
| Tiempo de inferencia | < 5 s | < 2 s | — |

### Métricas de Impacto

| Indicador | Objetivo |
|------------|-----------|
| Reducción de tiempo diagnóstico | 40% |
| Disminución de biopsias innecesarias | 25% |
| Incremento en detección temprana | 30% |
| Satisfacción de radiólogos | > 8.5/10 |

---

## Conclusión

La planificación presentada establece un marco claro para ejecutar un proyecto de inteligencia artificial aplicado al diagnóstico médico en un periodo de cuatro semanas. La estructura ágil propuesta, junto con una gestión eficiente de recursos y métricas bien definidas, asegura la viabilidad técnica, ética y académica del proyecto. Este enfoque permite entregar resultados medibles, escalables y con impacto clínico real en entornos hospitalarios de bajos recursos.
