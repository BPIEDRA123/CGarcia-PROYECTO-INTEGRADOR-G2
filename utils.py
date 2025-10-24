"""
utils.py - Funciones auxiliares extraídas de Integrador31.ipynb

Incluye utilidades generales como manejo de fechas/tiempo, archivos, paths,
logging sencillo y visualizaciones rápidas si las hubiera en el notebook.
Auto-generado para separar helpers reutilizables del resto del código.
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


# Constantes globales (si algunas son necesarias por utilidades)
SEED = 42
BASE_PATH = "/content/drive/MyDrive/p_1_image"
CLASSES = ["malignant", "benign", "normal"]
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_IMAGES_PER_CLASS = 10000


# Funciones auxiliares
def obtener_fecha_hora_actual():
    try:
        fecha_hora_actual = datetime.now()
        return fecha_hora_actual.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"⚠️ Error obteniendo hora del sistema: {e}")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

__all__ = ['obtener_fecha_hora_actual']
