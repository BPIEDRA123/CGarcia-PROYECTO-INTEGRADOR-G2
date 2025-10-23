
# Análisis del Script y EDA de Datos (Versión Markdown)

## 1. Lectura, análisis y razonamiento del script `Integrador31.ipynb`

**Resumen estructural**
- Número de celdas de código: 1
- Número de celdas de Markdown: 0
- El cuaderno concentra la lógica en una única celda extensa con múltiples importaciones y utilidades.

**Librerías detectadas (principales)**
- `cv2` (OpenCV): procesamiento de imágenes (lectura, conversión a escala de grises, redimensionamiento, filtros).
- `numpy`: operaciones vectorizadas y manejo de tensores.
- `plotly`: generación de visualizaciones interactivas.
- `imblearn` (SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTEENN, SMOTETomek): estrategias para manejo de desbalance de clases.
- Librerías estándar: `os`, `sys`, `time`, `traceback`, `warnings`.

**Inferencias sobre el objetivo del script**
- Procesamiento de imágenes ecográficas (uso de `cv2`).
- Preparación para aprendizaje automático con datos **desbalanceados** (SMOTE y variantes).
- Visualización de resultados y distribuciones (Plotly).
- Pipeline de preprocesamiento y posiblemente extracción de *features* explícitas o cálculo de métricas de calidad (nitidez, tamaño, intensidades), típico en proyectos de clasificación de imágenes médicas.

**Observaciones**
- El uso de técnicas de *oversampling* (SMOTE/ADASYN) sobre **imágenes** sugiere que el script probablemente trabaja con **vectores de características** previamente extraídos (p. ej., embeddings o descriptores), ya que aplicar SMOTE directamente en píxeles crudos de imágenes no es recomendable. Si se pretendiera balancear las clases a nivel de imagen, lo más idóneo es **data augmentation** específico (rotaciones, flips, cambios leves de brillo/contraste, recortes).


---

## 2. EDA (Exploratory Data Analysis) propuesto para el proyecto

> Nota de contexto: En este entorno **no se proporcionó el dataset de imágenes**; por tanto, se entrega un **EDA reproducible** en formato Markdown con los **indicadores clave, tablas esperadas y código de soporte**. Está alineado con el dataset indicado en la documentación del proyecto (e.g., *Thyroid_Ultrasound_2/p_image*). Cuando se coloquen las rutas correctas, los fragmentos de código producirán los resultados y figuras descritos.

### 2.1. Objetivos del EDA
1. Caracterizar el conjunto de datos de imágenes (tamaño, clases, distribución, resoluciones).
2. Evaluar calidad técnica (nitidez, ruido, contraste) y homogeneidad entre fuentes.
3. Detectar desbalanceo y proponer estrategias de mitigación (data augmentation).
4. Identificar posibles *data leaks* (duplicados/near-duplicados, mezclas de splits).
5. Establecer un *split* reproducible y métricas de base.

### 2.2. Inventario y estructura de datos
**Preguntas guía**
- ¿Cuántas imágenes hay por clase (`benigno`, `maligno`, `normal`)?
- ¿En qué formato se almacenan (PNG/JPG) y con qué resoluciones predominantes?
- ¿Existen metadatos asociados (edad, sexo, etc.)?

**Código de apoyo (Python)**
```python
from pathlib import Path
import cv2, numpy as np, pandas as pd

DATA_DIR = Path("/ruta/a/Thyroid_Ultrasound_2/p_image")  # ajustar
classes = ["benign", "malignant", "normal"]               # ajustar a nombres reales

rows = []
for cls in classes:
    for img_path in (DATA_DIR/cls).rglob("*.*"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        h, w = img.shape[:2]
        rows.append({"class": cls, "path": str(img_path), "height": h, "width": w})
df = pd.DataFrame(rows)
df.groupby("class")["path"].count().rename("count")
```

**Salida esperada**
- Tabla `conteo_por_clase` con el número de imágenes por clase.
- Tabla `resumen_resoluciones` con estadísticos por clase (min/mediana/max/ancho/alto).

### 2.3. Distribuciones (tamaño y aspecto)
**Indicadores**
- Histogramas de `width` y `height` por clase.
- Distribución del *aspect ratio* (`width/height`).

**Código**
```python
df["aspect_ratio"] = df["width"] / df["height"]
resol_summary = df.groupby("class")[["width","height","aspect_ratio"]].describe()
```

**Hallazgos esperables**
- Heterogeneidad de resoluciones (p. ej., 640×480 predominante, pero con variación).
- *Aspect ratios* cercanos a 1.33–1.78; si hay mucha dispersión conviene **estandarizar** (p. ej., *resize* a 224×224 con *padding*).

### 2.4. Calidad técnica de imagen
**Métricas**
- **Nitidez (Laplacian variance):** valores bajos sugieren desenfoque.
- **Contraste (std de intensidades):** distribución por clase.
- **Intensidad media:** para detectar diferencias sistemáticas por clase (potencial *leak*).

**Código**
```python
def sharpness_score(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def intensity_stats(img):
    return float(img.mean()), float(img.std())

stats = []
for p in df["path"]:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    s = sharpness_score(img)
    m, sd = intensity_stats(img)
    stats.append({"path": p, "sharpness": s, "mean_intensity": m, "std_intensity": sd})
q = pd.DataFrame(stats)
df = df.merge(q, on="path")
df.groupby("class")[["sharpness","mean_intensity","std_intensity"]].describe()
```

**Interpretación**
- Si una clase tiene **nitidez significativamente menor** (p. ej., malas adquisiciones), considerar normalización o filtros suaves (bilateral, CLAHE).
- Diferencias grandes de **intensidad media** entre clases podrían indicar **sesgo de adquisición**; mitigar con normalización y aumento de datos.

### 2.5. Duplicados y *near-duplicates*
**Estrategia**
- Usar *hash perceptual* (pHash/aHash/dHash) para detectar imágenes duplicadas o casi idénticas.
- Evitar que duplicados queden repartidos entre *train* y *test* (evita *data leakage*).

**Código**
```python
import imagehash
from PIL import Image

def phash(path):
    return imagehash.phash(Image.open(path))

df["phash"] = df["path"].apply(phash)
dups = df[df.duplicated("phash", keep=False)].sort_values("phash")
```

### 2.6. Desbalance de clases
**Diagnóstico**
- Calcular proporción por clase y **Gini** de distribución.

**Acciones**
- Para imágenes: **data augmentation** estratificado por clase (rotaciones pequeñas, flips horizontales, *random crop*, leves ajustes de brillo/contraste).
- Evitar SMOTE sobre píxeles crudos; si se usan embeddings/tablas, entonces sí se puede considerar SMOTE/ADASYN.

### 2.7. *Split* reproducible
**Reglas**
- `train/val/test = 70/20/10` estratificado por clase.
- Asegurar que **paciente/estudio** (si aplica) no se repita en *splits* distintos.
- Fijar `random_state` para reproducibilidad.

**Código**
```python
from sklearn.model_selection import StratifiedShuffleSplit
X = df["path"].values
y = df["class"].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
trainval_idx, test_idx = next(sss.split(X, y))
X_trainval, X_test = X[trainval_idx], X[test_idx]
y_trainval, y_test = y[trainval_idx], y[test_idx]
```

### 2.8. Tablas esperadas del EDA
- **Conteo por clase**: número absoluto y porcentaje.
- **Resoluciones por clase**: mediana, IQR, mínimo, máximo.
- **Calidad técnica**: mediana de nitidez y contraste por clase.
- **Duplicados**: lista de rutas o porcentajes afectados.
- **Sugerencias de *augmentation* por clase** según hallazgos.

### 2.9. Recomendaciones
1. **Estandarizar resolución** (p. ej., 224×224) con *padding* para preservar el aspecto.
2. **Normalización de intensidades** y aplicación de **CLAHE** para mejorar contraste.
3. **Data augmentation** focalizada en la(s) clase(s) minoritaria(s).
4. **Control de duplicados** antes de crear los *splits*.
5. Si se van a aplicar **SMOTE/ADASYN**, hacerlo sobre **representaciones tabulares** (p. ej., embeddings) y nunca sobre píxeles crudos.
6. Instrumentar un **pipeline reproducible** (scripts o notebooks modulares) y registrar métricas del EDA como parte de QA de datos.

---

## 3. Conclusiones
El script sugiere un flujo centrado en imágenes ecográficas con necesidades claras de manejo de desbalance y visualización de resultados. El EDA propuesto ofrece una ruta reproducible para caracterizar la calidad y estructura del conjunto de datos, mitigar riesgos de *data leakage* y establecer bases técnicas sólidas para el modelado posterior. Con los datos disponibles en la ruta indicada, los fragmentos de código producirán tablas y métricas objetivas para la toma de decisiones.
