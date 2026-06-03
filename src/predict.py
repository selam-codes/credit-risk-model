"""
Inference utilities — load model from MLflow registry and score new customers.
"""

import logging
from pathlib import Path
from typing import Union

import mlflow.sklearn
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "credit-risk-best-model"
MODEL_STAGE = "None"   # "Staging" or "Production" once promoted


def load_model(model_name: str = MODEL_NAME, stage: str = MODEL_STAGE):
    """Load the latest version of a registered model from MLflow."""
    uri = f"models:/{model_name}/latest"
    logger.info("Loading model from: %s", uri)
    model = mlflow.sklearn.load_model(uri)
    return model


def predict_risk(
    features: Union[dict, pd.DataFrame],
    model=None,
) -> dict:
    """
    Return risk probability and binary label for one or more customers.

    Args:
        features: dict of feature values or a pre-built DataFrame row(s).
        model: pre-loaded sklearn Pipeline (loaded once and reused in API).

    Returns:
        dict with 'risk_probability' and 'is_high_risk' (1/0).
    """
    if model is None:
        model = load_model()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features.copy()

    prob = model.predict_proba(df)[:, 1]
    label = (prob >= 0.5).astype(int)

    if len(prob) == 1:
        return {"risk_probability": float(prob[0]), "is_high_risk": int(label[0])}

    return {
        "risk_probability": prob.tolist(),
        "is_high_risk": label.tolist(),
    }


def credit_score_from_probability(risk_prob: float,
                                   score_min: int = 300,
                                   score_max: int = 850) -> int:
    """
    Convert a risk probability [0,1] into a credit score [score_min, score_max].
    Higher risk probability → lower credit score.
    """
    score = score_max - risk_prob * (score_max - score_min)
    return int(round(score))


if __name__ == "__main__":
    sample = {
        "total_amount": 15000.0,
        "avg_amount": 3000.0,
        "transaction_count": 5,
        "std_amount": 1200.0,
        "recency": 30,
        "frequency": 5,
        "monetary": 15000.0,
        "transaction_hour": 14,
        "transaction_day": 15,
        "transaction_month": 6,
        "transaction_year": 2025,
        "ProductCategory": "airtime",
        "ChannelId": "ChannelId_2",
        "ProviderId": "ProviderId_1",
    }

    model = load_model()
    result = predict_risk(sample, model=model)
    result["credit_score"] = credit_score_from_probability(result["risk_probability"])
    print(result)
