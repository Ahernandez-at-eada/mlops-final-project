"""
test_preprocessing.py

Tests unitarios básicos para el módulo de preprocesamiento. Se
ejecutan en el pipeline de CI (on pull request) para detectar
regresiones antes de hacer merge a main.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (   # noqa: E402
    align_columns,
    clean_data,
    encode_features,
)


def make_sample_df():
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB"],
            "gender": ["Female", "Male"],
            "tenure": [1, 34],
            "Contract": ["Month-to-month", "One year"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [" ", "1889.5"],
            "Churn": ["No", "Yes"],
        }
    )


def test_clean_data_drops_customer_id():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns


def test_clean_data_handles_blank_total_charges():
    df = make_sample_df()
    cleaned = clean_data(df)
    # La fila con TotalCharges=' ' debe quedar imputada a 0, no NaN
    assert cleaned.loc[0, "TotalCharges"] == 0
    assert cleaned["TotalCharges"].dtype.kind in "if"


def test_encode_features_returns_binary_target():
    df = make_sample_df()
    cleaned = clean_data(df)
    X, y, feature_columns = encode_features(cleaned)
    assert set(y.unique()).issubset({0, 1})
    assert "Churn" not in X.columns
    assert len(feature_columns) == X.shape[1]


def test_align_columns_fills_missing_with_zero():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    aligned = align_columns(df, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert (aligned["c"] == 0).all()


def test_align_columns_drops_extra_columns():
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [99]})
    aligned = align_columns(df, ["a", "b"])
    assert list(aligned.columns) == ["a", "b"]
