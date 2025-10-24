"""
data_processing.py - Funciones de procesamiento extraídas de Integrador31.ipynb

Este módulo contiene funciones utilitarias para carga, preprocesamiento, 
extracción de características y construcción de datasets.
Auto-generado para separar la lógica de procesamiento del resto del código.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
from scipy import stats, ndimage
from scipy.ndimage import sobel, gaussian_filter
from scipy.stats import kurtosis, skew, shapiro, normaltest
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight, resample
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import PartialDependenceDisplay
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive, files
import warnings
from datetime import datetime
import pytz
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Constantes utilizadas
SEED = 42
BASE_PATH = "/content/drive/MyDrive/p_1_image"
CLASSES = ["malignant", "benign", "normal"]
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_IMAGES_PER_CLASS = 10000


# Funciones de procesamiento
def mejorar_calidad_imagen(imagen):
    try:
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(0.9)
        enhancer = ImageEnhance.Contrast(imagen)
        imagen = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Sharpness(imagen)
        imagen = enhancer.enhance(1.05)
        return imagen
    except Exception as e:
        print(f"⚠️ Error mejorando calidad de imagen: {e}")
        return imagen


def cargar_y_preprocesar_imagen_avanzado(ruta, tamaño=IMG_SIZE):
    try:
        with Image.open(ruta) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = mejorar_calidad_imagen(img)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.SMOOTH)
            img.thumbnail((tamaño[0] * 2, tamaño[1] * 2), Image.Resampling.LANCZOS)
            img = img.resize(tamaño, Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.power(arr, 1.1)
            if arr.shape != (*tamaño, 3):
                import cv2
                arr = cv2.resize(arr, tamaño)
            return arr
    except Exception as e:
        print(f"❌ Error avanzado procesando {ruta}: {e}")
        return None


def es_archivo_imagen_avanzado(nombre):
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
    return (nombre.lower().endswith(extensiones_validas) and
            not nombre.startswith('.') and
            os.path.isfile(nombre))


def extraer_caracteristicas_avanzadas_completas(arr):
    try:
        import cv2
        gris = np.mean(arr, axis=2)
        intensidad = np.mean(gris)
        contraste = np.std(gris)
        entropia = stats.entropy(gris.flatten() + 1e-8)
        momentos = cv2.moments((gris * 255).astype(np.uint8))
        hu_momentos = cv2.HuMoments(momentos).flatten()
        bordes_canny = cv2.Canny((gris * 255).astype(np.uint8), 30, 150)
        densidad_bordes = np.mean(bordes_canny > 0)
        grad_x = sobel(gris, axis=0)
        grad_y = sobel(gris, axis=1)
        magnitud_gradiente = np.sqrt(grad_x**2 + grad_y**2)
        asimetria = skew(gris.flatten())
        curtosis = kurtosis(gris.flatten())

        return {
            'intensidad_promedio': float(intensidad),
            'contraste': float(contraste),
            'entropia': float(entropia),
            'asimetria': float(asimetria),
            'curtosis': float(curtosis),
            'densidad_bordes': float(densidad_bordes),
            'magnitud_gradiente_promedio': float(np.mean(magnitud_gradiente)),
            'hu_momento_1': float(hu_momentos[0]),
            'hu_momento_2': float(hu_momentos[1]),
            'heterogeneidad': float(contraste / (intensidad + 1e-8))
        }
    except Exception as e:
        print(f"⚠️ Error en análisis avanzado: {e}")
        return {
            'intensidad_promedio': 0.5, 'contraste': 0.2, 'entropia': 0.0,
            'asimetria': 0.0, 'curtosis': 0.0, 'densidad_bordes': 0.05,
            'magnitud_gradiente_promedio': 0.1, 'hu_momento_1': 0.0,
            'hu_momento_2': 0.0, 'heterogeneidad': 0.4
        }


def cargar_dataset_completo_avanzado():
    import cv2
    todas_imagenes = []
    todas_etiquetas = []
    todos_metadatos = []
    estadisticas_carga = {clase: 0 for clase in CLASSES}

    for clase in CLASSES:
        ruta_clase = os.path.join(BASE_PATH, clase)
        if not os.path.exists(ruta_clase):
            print(f"⚠️ Carpeta no encontrada: {ruta_clase}")
            continue

        archivos = sorted([f for f in os.listdir(ruta_clase)
                          if es_archivo_imagen_avanzado(os.path.join(ruta_clase, f))])

        print(f"\n📂 PROCESANDO {clase.upper()}: {len(archivos)} imágenes encontradas")

        for i, archivo in enumerate(archivos[:MAX_IMAGES_PER_CLASS]):
            ruta_completa = os.path.join(ruta_clase, archivo)
            if i % 100 == 0 and i > 0:
                print(f"   🚀 Procesadas {i}/{min(len(archivos), MAX_IMAGES_PER_CLASS)} imágenes...")

            imagen = cargar_y_preprocesar_imagen_avanzado(ruta_completa)

            if imagen is not None and imagen.shape == (*IMG_SIZE, 3):
                todas_imagenes.append(imagen)
                todas_etiquetas.append(clase)
                caracteristicas = extraer_caracteristicas_avanzadas_completas(imagen)

                try:
                    with Image.open(ruta_completa) as img:
                        ancho, alto = img.size
                        dimensiones_numericas = f"{ancho}x{alto}"
                except:
                    ancho, alto = IMG_SIZE
                    dimensiones_numericas = f"{ancho}x{alto}"

                metadato = {
                    'clase': clase,
                    'archivo': archivo,
                    'ruta': ruta_completa,
                    'tamaño_kb': os.path.getsize(ruta_completa) / 1024,
                    'dimensiones_originales': dimensiones_numericas,
                    'ancho_original': ancho,
                    'alto_original': alto,
                    'procesado_exitoso': True,
                    **caracteristicas
                }
                todos_metadatos.append(metadato)
                estadisticas_carga[clase] += 1

    if len(todas_imagenes) == 0:
        print("❌ No se pudieron cargar imágenes. Creando dataset de ejemplo...")
        return crear_dataset_ejemplo()

    X = np.array(todas_imagenes, dtype=np.float32)
    y = np.array(todas_etiquetas)
    df_metadatos = pd.DataFrame(todos_metadatos)

    print(f"\n✅ CARGA AVANZADA COMPLETADA:")
    print(f"   • Total imágenes procesadas: {len(X):,}")
    print(f"   • Dimensiones del dataset: {X.shape}")
    print(f"   • Memoria utilizada: {X.nbytes / (1024**3):.2f} GB")

    print(f"   • Distribución por clase:")
    for clase, count in estadisticas_carga.items():
        if count > 0:
            print(f"     {clase}: {count:,} imágenes")

    return X, y, df_metadatos


def crear_dataset_ejemplo():
    print("🔧 Creando dataset de ejemplo para pruebas...")
    todas_imagenes = []
    todas_etiquetas = []
    metadatos = []

    for i in range(300):
        img = np.random.normal(0.5, 0.2, (IMG_SIZE[0], IMG_SIZE[1], 3))
        img = np.clip(img, 0, 1).astype(np.float32)
        clase = CLASSES[i % len(CLASSES)]
        todas_imagenes.append(img)
        todas_etiquetas.append(clase)

        metadatos.append({
            'clase': clase,
            'archivo': f'ejemplo_{i}.jpg',
            'ruta': f'/synthetic/ejemplo_{i}.jpg',
            'tamaño_kb': 250.0,
            'dimensiones_originales': f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
            'ancho_original': IMG_SIZE[0],
            'alto_original': IMG_SIZE[1],
            'procesado_exitoso': True,
            'intensidad_promedio': 0.5 + (i % 3) * 0.1,
            'contraste': 0.2 + (i % 3) * 0.05,
            'entropia': 2.0 + (i % 3) * 0.3,
            'asimetria': 0.1 * (i % 3),
            'curtosis': -0.5 + (i % 3) * 0.2,
            'densidad_bordes': 0.05 + (i % 3) * 0.02,
            'magnitud_gradiente_promedio': 0.1 + (i % 3) * 0.05,
            'hu_momento_1': 0.2 + (i % 3) * 0.1,
            'hu_momento_2': 0.1 + (i % 3) * 0.05,
            'heterogeneidad': 0.4 + (i % 3) * 0.1
        })

    return np.array(todas_imagenes), np.array(todas_etiquetas), pd.DataFrame(metadatos)

__all__ = ['mejorar_calidad_imagen', 'cargar_y_preprocesar_imagen_avanzado', 'es_archivo_imagen_avanzado', 'extraer_caracteristicas_avanzadas_completas', 'cargar_dataset_completo_avanzado', 'crear_dataset_ejemplo']
