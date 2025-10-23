# Descripción de los datos

## Descripción general
Este directorio almacena los datos utilizados en el proyecto **“Sistema de apoyo al diagnóstico basado en IA para detección temprana de cáncer de tiroides mediante ecografía”**.

- `raw/`: datos originales (sin modificaciones).  
- `processed/`: datos limpios y/o transformados listos para entrenamiento, validación y prueba.

## Estructura esperada
```
data/
├─ raw/
│  ├─ images/                # Ecografías originales (si aplica)
│  └─ metadata.csv           # Metadatos/etiquetas originales
└─ processed/
   ├─ train/                 # División de entrenamiento (imágenes o arrays)
   ├─ val/                   # División de validación
   ├─ test/                  # División de prueba
   ├─ features.csv           # Features tabulares (si se extrajeron)
   └─ manifest.csv           # Inventario con ruta, clase y split
```

## Convenciones y estándares
- Formato de imagen recomendado: PNG/JPG en escala de grises.  
- Resolución estandarizada: **224×224** con *padding* para preservar aspecto.  
- Nombres de clase: `benign`, `malignant`, `normal` (ajustar según tu dataset).  
- Normalización de intensidades en `[0,1]` o z-score por imagen.

## Diccionario de campos (archivos tabulares)
| Campo           | Tipo    | Descripción                                                       |
|-----------------|---------|-------------------------------------------------------------------|
| `path`          | string  | Ruta relativa al archivo de imagen.                               |
| `class`         | string  | Etiqueta de la imagen (p. ej., `benign`/`malignant`/`normal`).    |
| `width`         | int     | Ancho en píxeles.                                                 |
| `height`        | int     | Alto en píxeles.                                                  |
| `aspect_ratio`  | float   | Relación `width/height`.                                          |
| `sharpness`     | float   | Varianza del Laplaciano (nitidez).                                |
| `mean_intensity`| float   | Intensidad media (0–255 si 8-bit).                                |
| `std_intensity` | float   | Desviación estándar de intensidad.                                |
| `split`         | string  | `train` / `val` / `test`.                                         |

## Buenas prácticas
- Mantener **`raw/` inmutable**; todo cambio va a `processed/`.  
- Versionar `manifest.csv` y `features.csv`.  
- Registrar el **script o notebook** que generó cada artefacto de `processed/`.  
- Evitar subir datos sensibles a GitHub (aplica anonimización y revisa licencias).

## Notas de licencia
Los datos y recursos de terceros conservan sus licencias originales. Revisa y respeta las condiciones de uso antes de redistribuir.
