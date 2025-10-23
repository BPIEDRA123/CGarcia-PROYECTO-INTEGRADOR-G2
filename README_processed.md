# Datos Procesados (processed)

Este directorio contiene los datos derivados o transformados a partir de los archivos originales en `data/raw/`.

## Contenido esperado
- Imágenes preprocesadas (224×224 px, escala de grises).
- Features tabulares (`features.csv`) con estadísticas de imagen (contraste, intensidad, nitidez, etc.).
- Manifestos o divisiones (`manifest.csv`) para entrenamiento, validación y test.
- Divisiones por carpetas (`train/`, `val/`, `test/`).

## Buenas prácticas
1. Mantener trazabilidad entre archivos de `raw` y `processed` (mismo ID o nombre base).
2. Documentar el script o notebook que generó cada versión del dataset.
3. Guardar metadatos de procesamiento (fecha, parámetros usados, versión del código).
4. No sobrescribir versiones anteriores; usar subcarpetas por fecha o versión si se vuelve a procesar.

## Ejemplo de estructura
```
data/processed/
├─ train/
│  ├─ benign/
│  └─ malignant/
├─ val/
├─ test/
├─ features.csv
└─ manifest.csv
```
