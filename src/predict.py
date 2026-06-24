"""
predict.py

Workflow 'on-demand': toma el dataset en
batch_prediction_dataset/on_demand_dataset.csv, aplica el modelo ya
entrenado (models/model.pkl) y guarda los resultados en el mismo
directorio.

Uso:
    python src/predict.py
"""

import json
import os

import joblib
import pandas as pd

from preprocessing import align_columns, clean_data

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

BATCH_DIR = "batch_prediction_dataset"
INPUT_PATH = os.path.join(BATCH_DIR, "on_demand_dataset.csv")
OUTPUT_PATH = os.path.join(BATCH_DIR, "predictions.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecuta primero src/train.py."
        )
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset de entrada en {INPUT_PATH}."
        )

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)

    df_raw = pd.read_csv(INPUT_PATH)

    # Guardamos el customerID si existe, para reportarlo junto al resultado
    customer_ids = (
        df_raw["customerID"] if "customerID" in df_raw.columns else None
    )

    df_clean = clean_data(df_raw)

    # Si el dataset on-demand trae la columna Churn (caso de prueba),
    # la quitamos antes de codificar para no usarla como feature.
    if "Churn" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Churn"])

    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    X_new = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    X_new = align_columns(X_new, feature_columns)

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]

    output = pd.DataFrame(
        {
            "churn_prediction": ["Yes" if p == 1 else "No" for p in predictions],
            "churn_probability": probabilities.round(4),
        }
    )
    if customer_ids is not None:
        output.insert(0, "customerID", customer_ids)

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Predicciones guardadas en: {OUTPUT_PATH}")
    print(output.head())


if __name__ == "__main__":
    main()
