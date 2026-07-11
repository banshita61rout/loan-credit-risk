"""
train_model.py
----------------
Trains the 3 models from the notebook (Logistic Regression, KNN, Naive Bayes)
using a proper sklearn Pipeline (imputation + encoding + scaling baked in),
so the saved model can take raw applicant data directly at prediction time.

Run this once locally:
    python train_model.py

It produces:
    models/logistic_regression.pkl
    models/knn.pkl
    models/naive_bayes.pkl
    models/metrics.json
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "loan_approval_dataset.csv"
MODEL_DIR = "models"

NUMERIC_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

CATEGORICAL_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

TARGET_COL = "Loan_Approved"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the feature engineering done in the notebook (Credit_Score^2, DTI_Ratio^2)."""
    df = df.copy()
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS + ["DTI_Ratio_sq", "Credit_Score_sq"]),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def evaluate(name, y_test, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label="Yes"), 4),
        "recall": round(recall_score(y_test, y_pred, pos_label="Yes"), 4),
        "f1": round(f1_score(y_test, y_pred, pos_label="Yes"), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Drop rows with no target label -- can't train on unknown outcomes
    df = df.dropna(subset=[TARGET_COL])

    # Applicant_ID is just a row identifier, not predictive
    df = df.drop(columns=["Applicant_ID"])

    df = add_engineered_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=13),
        "naive_bayes": GaussianNB(),
    }

    all_metrics = []

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate(name, y_test, y_pred)
        all_metrics.append(metrics)
        print(f"--- {name} ---")
        print(metrics)

        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved 3 model pipelines + metrics.json to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
