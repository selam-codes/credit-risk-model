# Credit Risk Probability Model for Alternative Data

An end-to-end implementation for building, deploying, and automating a credit risk model for Bati Bank's buy-now-pay-later service, using transaction data from an eCommerce partner.

## Credit Scoring Business Understanding

### How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Capital Accord requires banks to quantify credit risk using internal models, but mandates that these models be transparent, validated, and explainable to regulators. Under Pillar 1 (Minimum Capital Requirements), banks must demonstrate that their probability-of-default (PD) estimates are derived from sound statistical methods. Under Pillar 2 (Supervisory Review), regulators must be able to audit the model logic, inspect feature contributions, and trace every scoring decision back to documented assumptions.

This creates a direct tension with black-box approaches: a model that cannot be interrogated cannot be approved. Practically, this means:

- **Logistic Regression with WoE-encoded features** is often preferred in regulated credit scoring because the model coefficients map directly to scorecard points, every feature's contribution is additive and auditable, and the output is a calibrated probability.
- **Documentation requirements** extend beyond code — each modeling choice (feature selection rationale, binning strategy, proxy variable justification) must be written up as a formal model development document (MDD).
- **Monitoring and recalibration** must be planned from day one; Basel II expects evidence that the model remains valid over time (population stability, Gini drift).

### Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The raw eCommerce transaction dataset contains no ground-truth default outcome — customers have never been extended credit, so there is no repayment history to observe. A supervised model requires a binary target. A **proxy variable** bridges this gap by inferring default risk from observable behavioral signals.

The chosen proxy is built on **RFM (Recency, Frequency, Monetary) analysis**: customers who have low transaction frequency, low monetary value, and have been inactive for a long time are treated as disengaged and labeled high-risk (potential defaulters). This mirrors real-world credit bureau logic, where thin-file and inactive consumers are considered higher risk.

**Business risks introduced by proxy-based prediction:**

| Risk | Description |
|------|-------------|
| **Label noise** | The proxy may misclassify genuinely good customers who simply didn't transact recently (e.g., seasonal buyers) as high-risk. |
| **Concept drift** | The RFM proxy captures eCommerce engagement, not loan repayment behavior. A customer who buys frequently may still default on a loan. |
| **Regulatory scrutiny** | Regulators may challenge the proxy's validity — the model sponsor must document why RFM disengagement is a reasonable stand-in for default. |
| **Feedback loops** | If the model denies credit to proxy-labeled high-risk customers, we never observe their true repayment behavior, preventing future model improvement. |
| **Bias risk** | RFM metrics can correlate with demographic attributes; the proxy may introduce disparate impact against protected groups. |

### What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|-----------|--------------------------|--------------------------------------|
| **Interpretability** | High — scorecard points, additive contributions | Low — requires SHAP or LIME post-hoc |
| **Regulatory acceptance** | Directly supported by Basel II guidance | Requires additional explainability layer |
| **Performance (AUC)** | Moderate (~0.70–0.78 on typical credit data) | High (~0.78–0.85+) |
| **Feature handling** | Requires manual binning and WoE encoding | Handles non-linearity and interactions natively |
| **Overfitting risk** | Low with proper regularization | Higher; requires careful tuning and cross-validation |
| **Audit trail** | Clear — coefficient × WoE = log-odds contribution | Opaque without SHAP; not always accepted by regulators |
| **Deployment** | Simple scoring formula, runs anywhere | Requires model serialization and inference infrastructure |
| **Monitoring** | PSI on WoE bins is standard practice | Requires monitoring feature drift + prediction drift |

**Recommendation for this project:** Train both models, compare on AUC/Gini and F1, but default to the interpretable model for the production scorecard unless the performance gap exceeds 3–5 Gini points. If gradient boosting is chosen, document SHAP global importances and include them in the model development document.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # add to .gitignore
│   ├── raw/                       # Raw data (xente_data.csv)
│   └── processed/                 # Processed data for training
├── notebooks/
│   └── eda.ipynb                  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training + MLflow tracking
│   ├── predict.py                 # Inference utilities
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place the raw data file at data/raw/data.csv

# Run feature engineering + proxy label creation
python src/data_processing.py

# Train models (tracked in MLflow)
python src/train.py

# Start the API
uvicorn src.api.main:app --reload

# Or use Docker
docker-compose up --build
```

## Data

Dataset: [Xente Challenge on Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)

Place the downloaded CSV at `data/raw/data.csv`.

## Running Tests

```bash
pytest tests/ -v
```

## API Usage

Once running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

**POST /predict**
```json
{
  "total_amount": 15000.0,
  "avg_amount": 3000.0,
  "transaction_count": 5,
  "std_amount": 1200.0,
  "recency_days": 30,
  "transaction_hour": 14,
  "transaction_day": 15,
  "transaction_month": 6,
  "transaction_year": 2025,
  "product_category": "airtime",
  "channel_id": "ChannelId_2",
  "provider_id": "ProviderId_1"
}
```

## MLflow Tracking

```bash
mlflow ui
```

Visit `http://localhost:5000` to compare experiment runs.
