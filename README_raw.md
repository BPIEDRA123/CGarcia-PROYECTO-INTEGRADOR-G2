# Datos Originales (raw)

Este directorio contiene los **datos originales** del proyecto antes de cualquier tipo de procesamiento, limpieza o normalización.

## Contenido esperado
- Imágenes ecográficas en formato **PNG o JPG**.
- Metadatos o etiquetas asociados en un archivo **CSV** (`metadata.csv`), con campos como:
  - `image_id`
  - `class` (`benign`, `malignant`, `normal`)
  - `source` (hospital, repositorio, etc.)
  - `license` (fuente o tipo de licencia)

## Reglas y buenas prácticas
1. No modificar ni renombrar los archivos originales.
2. No alterar los valores del CSV original.
3. Documentar toda nueva fuente de datos agregada.
4. Conservar la estructura de carpetas original por lote o repositorio.
5. Evitar subir datos personales o sensibles (mantener anonimización).

## Ejemplo de estructura
```
data/raw/
├─ images/
│  ├─ benign_001.png
│  ├─ malignant_002.png
│  └─ normal_003.png
└─ metadata.csv
```
