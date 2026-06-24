"""
train.py

Entrena el modelo de clasificación de churn, registra experimentos en
MLflow y guarda el modelo final en models/ junto con la lista de
columnas de features (necesaria para el workflow on-demand).

Uso:
    python src/train.py

Esto se ejecuta tanto manualmente (durante experimentación, ver
notebooks/) como automáticamente en el pipeline de CD (on push a main).
"""

import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    clean_data,
    encode_features,
    load_raw_data,
    split_data,
)

DATA_PATH = os.path.join("datasets", "telco_churn.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

MLFLOW_EXPERIMENT_NAME = "telco-churn-prediction"


def evaluate(model, X_test, y_test) -> dict:
    """Calcula múltiples métricas de desempeño sobre el set de test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def get_candidate_models() -> dict:
    """
    Modelos candidatos a evaluar. Se mantienen intencionalmente
    simples (sin grandes búsquedas de hiperparámetros) siguiendo la
    recomendación del enunciado de evitar modelos sobrecomplicados.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42
        ),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 1. Carga y limpieza
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    X, y, feature_columns = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model = None
    best_model_name = None
    best_f1 = -1.0
    results_summary = []

    # 2. Entrenar y registrar cada modelo candidato como un run de MLflow
    for name, model in get_candidate_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model_type", name)
            safe_params = {
                k: v
                for k, v in model.get_params().items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            mlflow.log_params(safe_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            results_summary.append({"model": name, **metrics})
            print(f"[{name}] {metrics}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_model_name = name

    # 3. Guardar el mejor modelo para el resto del pipeline (CD / on-demand)
    joblib.dump(best_model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_columns, f)

    print(f"\nMejor modelo: {best_model_name} (f1={best_f1:.4f})")
    print(f"Modelo guardado en: {MODEL_PATH}")

    # Resumen en csv para referencia rápida sin entrar a la UI de MLflow
    pd.DataFrame(results_summary).to_csv(
        os.path.join(MODELS_DIR, "training_results.csv"), index=False
    )


if __name__ == "__main__":
    main()
