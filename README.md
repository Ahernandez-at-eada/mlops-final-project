Telco Customer Churn Prediction
Project for the course MLOps and System Design (EADA Business School). Training and prediction pipeline for customer churn in a telecommunications company.

Project Structure
```
telco-churn-mlops/
├── main.py                          # Punto de entrada (train / predict)
├── requirements.txt
├── src/
│   ├── preprocessing.py             # Limpieza y encoding de datos
│   ├── train.py                     # Entrenamiento + tracking en MLflow
│   └── predict.py                   # Workflow on-demand de predicción
├── notebooks/
│   └── experimentation.ipynb        # Notebook de exploración y experimentos
├── datasets/
│   └── telco_churn.csv              # Dataset original (IBM Telco Churn)
├── models/
│   ├── model.pkl                    # Modelo entrenado (generado por CD)
│   ├── feature_columns.json         # Columnas usadas en entrenamiento
│   └── training_results.csv         # Resumen de métricas por modelo
├── batch_prediction_dataset/
│   ├── on_demand_dataset.csv        # Input para el workflow on-demand
│   └── predictions.csv              # Output generado por predict.py
├── tests/
│   └── test_preprocessing.py        # Tests unitarios (corridos en CI)
└── .github/workflows/
    ├── ci.yml                       # CI: tests en cada pull request
    └── cd.yml                       # CD: entrena el modelo en push a main
```

## How to Run

```bash
pip install -r requirements.txt

# Entrenar el modelo (registra experimentos en MLflow)
python main.py train

# Ver resultados en la UI de MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Correr el workflow on-demand de predicción
python main.py predict

# Correr los tests
pytest tests/ -v
```

## Problem and Model Type
Binary classification: predict whether a customer will churn (Churn: Yes/No) based on their demographic data, contract, and subscribed services.
## CI/CD

- CI (ci.yml): runs on every pull request to main, executing the test suite from tests/.

-CD (cd.yml): runs on every push to main, retrains the model from scratch and saves it in models/.

## Workflow on-demand

src/predict.py takes batch_prediction_dataset/on_demand_dataset.csv, applies the trained model, and saves the results in batch_prediction_dataset/predictions.csv
