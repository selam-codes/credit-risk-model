"""
Model training, hyperparameter tuning, and MLflow tracking.

Usage:
    python src/train.py
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/features.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "credit-risk-model"

NUMERICAL_FEATURES = [
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount",
    "recency",
    "frequency",
    "monetary",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
]

CATEGORICAL_FEATURES = [
    "ProductCategory",
    "ChannelId",
    "ProviderId",
]

TARGET = "is_high_risk"


def load_data(path: Path = PROCESSED_PATH):
    df = pd.read_csv(path)
    available_features = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[available_features]
    y = df[TARGET]
    logger.info("Dataset: %d samples, %d features, %.2f%% high-risk",
                len(y), len(available_features), y.mean() * 100)
    return X, y


def build_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def train_logistic_regression(X_train, y_train, preprocessor) -> Pipeline:
    param_dist = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__max_iter": [200, 500],
    }
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=10, cv=5, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("LR best params: %s", search.best_params_)
    return search.best_estimator_


def train_random_forest(X_train, y_train, preprocessor) -> Pipeline:
    param_dist = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        )),
    ])
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=10, cv=5, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("RF best params: %s", search.best_params_)
    return search.best_estimator_


def train_gradient_boosting(X_train, y_train, preprocessor) -> Pipeline:
    param_dist = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.7, 0.8, 1.0],
    }
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=10, cv=5, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info("GB best params: %s", search.best_params_)
    return search.best_estimator_


def run_experiment(name: str, model_fn, X_train, y_train, X_test, y_test, preprocessor):
    with mlflow.start_run(run_name=name):
        logger.info("Training %s ...", name)
        model = model_fn(X_train, y_train, preprocessor)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(model.named_steps["classifier"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            "%s  AUC=%.4f  F1=%.4f  Acc=%.4f",
            name, metrics["roc_auc"], metrics["f1"], metrics["accuracy"],
        )
        return model, metrics


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    min_class = y.value_counts().min()
    stratify = y if min_class >= 2 else None
    if stratify is None:
        logger.warning("Too few samples in minority class (%d) for stratified split — falling back to random split.", min_class)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    preprocessor = build_preprocessor(X_train)

    experiments = [
        ("LogisticRegression", train_logistic_regression),
        ("RandomForest", train_random_forest),
        ("GradientBoosting", train_gradient_boosting),
    ]

    results = {}
    for name, fn in experiments:
        model, metrics = run_experiment(name, fn, X_train, y_train, X_test, y_test, preprocessor)
        results[name] = (model, metrics)

    best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_model, best_metrics = results[best_name]
    logger.info("Best model: %s  AUC=%.4f", best_name, best_metrics["roc_auc"])

    # Register the best model in MLflow Model Registry
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = '{best_name}'",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs.empty:
        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri, "credit-risk-best-model")
        logger.info("Registered model from run %s as 'credit-risk-best-model'", best_run_id)


if __name__ == "__main__":
    main()
