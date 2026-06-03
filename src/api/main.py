"""
FastAPI credit risk scoring service.

Loads the best registered model from MLflow and exposes a /predict endpoint.
"""

import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import HealthResponse, PredictionRequest, PredictionResponse
from src.predict import credit_score_from_probability, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = load_model()
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model could not be loaded at startup: %s", exc)
        logger.warning("Call /predict after training and registering a model.")
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Predicts credit risk probability for new customers using an RFM-based "
        "proxy label and a trained scikit-learn Pipeline registered in MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Service liveness check."""
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict(request: PredictionRequest):
    """
    Score a new customer and return:
    - **risk_probability** — probability of being high-risk [0, 1]
    - **is_high_risk** — binary label (1 = high risk)
    - **credit_score** — integer score [300, 850]; higher = lower risk
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and register a model first (python src/train.py).",
        )

    features = pd.DataFrame([request.model_dump()])
    try:
        prob = float(_model.predict_proba(features)[0, 1])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    label = int(prob >= 0.5)
    score = credit_score_from_probability(prob)

    return PredictionResponse(
        risk_probability=round(prob, 6),
        is_high_risk=label,
        credit_score=score,
    )
