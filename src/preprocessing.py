"""
preprocessing.py

Funciones de carga y transformación de datos para el proyecto de
predicción de churn de clientes de telco.

Todas las transformaciones quedan centralizadas aquí para que el
notebook de experimentación, el pipeline de entrenamiento (CD) y el
workflow on-demand de predicción usen exactamente la misma lógica.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Columnas que no aportan señal predictiva (identificador único)
DROP_COLS = ["customerID"]

# Columna objetivo
TARGET_COL = "Churn"


def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el csv crudo del dataset de churn."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
    - TotalCharges viene como string con espacios en blanco en
      clientes con tenure=0 (clientes nuevos). Se convierte a numérico
      y se imputan esos casos con 0.
    - Se elimina el identificador de cliente.
    """
    df = df.copy()

    # TotalCharges tiene strings vacíos ' ' para tenure=0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen viene como 0/1 (int), lo dejamos igual
    # Quitamos columnas que no aportan
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def encode_features(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Codifica variables categóricas con one-hot encoding y separa
    features (X) de target (y). El target se mapea a binario 0/1.

    Devuelve (X, y, feature_columns) donde feature_columns es la lista
    de columnas finales usadas por el modelo (necesaria para alinear
    el dataset de predicción on-demand con las mismas columnas).
    """
    df = df.copy()

    y = df[target_col].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y, X.columns.tolist()


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split estratificado train/test (el churn suele estar desbalanceado)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def align_columns(X_new: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """
    Alinea un dataframe nuevo (p.ej. batch de predicción on-demand) con
    las columnas exactas usadas en entrenamiento. Columnas faltantes se
    rellenan con 0, columnas extra se descartan.
    """
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)
    return X_new
