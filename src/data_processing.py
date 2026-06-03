"""
Feature engineering pipeline and proxy target variable (is_high_risk) creation.

Run directly to produce data/processed/features.csv:
    python src/data_processing.py
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/data.csv")
PROCESSED_PATH = Path("data/processed/features.csv")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Compute per-customer aggregate statistics from transaction rows."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        agg = (
            X.groupby("CustomerId")["Amount"]
            .agg(
                total_amount="sum",
                avg_amount="mean",
                transaction_count="count",
                std_amount="std",
            )
            .reset_index()
        )
        agg["std_amount"] = agg["std_amount"].fillna(0)
        return agg


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Extract hour/day/month/year from TransactionStartTime."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        ts = pd.to_datetime(df["TransactionStartTime"], utc=True)
        df["transaction_hour"] = ts.dt.hour
        df["transaction_day"] = ts.dt.day
        df["transaction_month"] = ts.dt.month
        df["transaction_year"] = ts.dt.year
        return df


class RFMFeatures(BaseEstimator, TransformerMixin):
    """Compute Recency, Frequency, Monetary features per customer."""

    def __init__(self, snapshot_date: pd.Timestamp = None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        if self.snapshot_date is None:
            ts = pd.to_datetime(X["TransactionStartTime"], utc=True)
            self.snapshot_date_ = ts.max() + pd.Timedelta(days=1)
        else:
            self.snapshot_date_ = self.snapshot_date
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        ts = pd.to_datetime(df["TransactionStartTime"], utc=True)
        df["_ts"] = ts

        rfm = (
            df.groupby("CustomerId")
            .agg(
                recency=("_ts", lambda x: (self.snapshot_date_ - x.max()).days),
                frequency=("_ts", "count"),
                monetary=("Amount", "sum"),
            )
            .reset_index()
        )
        return rfm


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Simple Weight-of-Evidence encoder for categorical columns.
    Falls back gracefully when a column has only one class.
    """

    def __init__(self, columns: list, target_col: str = "is_high_risk"):
        self.columns = columns
        self.target_col = target_col
        self.woe_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        if y is None:
            return self

        total_events = y.sum()
        total_non_events = len(y) - total_events

        for col in self.columns:
            if col not in X.columns:
                continue
            woe_map = {}
            for cat, group in X.groupby(col):
                events = y[group.index].sum()
                non_events = len(group) - events
                dist_events = (events + 0.5) / (total_events + 0.5)
                dist_non_events = (non_events + 0.5) / (total_non_events + 0.5)
                woe_map[cat] = np.log(dist_events / dist_non_events)
            self.woe_maps_[col] = woe_map
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if col in df.columns:
                df[col + "_woe"] = df[col].map(woe_map).fillna(0)
                df.drop(columns=[col], inplace=True)
        return df


# ---------------------------------------------------------------------------
# RFM-based proxy label
# ---------------------------------------------------------------------------

def create_proxy_label(rfm_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    K-Means cluster customers on RFM, then label the least-engaged
    cluster as high-risk (is_high_risk = 1).

    Log-transforms frequency and monetary before scaling to prevent extreme
    monetary outliers from collapsing the clustering into a near-empty cluster.
    """
    features = rfm_df[["recency", "frequency", "monetary"]].copy()
    # Log-transform to reduce skewness — financial data is heavily right-tailed
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["cluster"] = kmeans.fit_predict(scaled)

    cluster_stats = (
        rfm_df.groupby("cluster")[["frequency", "monetary"]]
        .mean()
    )
    # Normalise each metric to [0,1] before summing so neither dominates
    for col in ["frequency", "monetary"]:
        col_range = cluster_stats[col].max() - cluster_stats[col].min()
        cluster_stats[col] = (cluster_stats[col] - cluster_stats[col].min()) / (col_range + 1e-9)
    cluster_stats["score"] = cluster_stats["frequency"] + cluster_stats["monetary"]
    high_risk_cluster = cluster_stats["score"].idxmin()

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)
    logger.info(
        "High-risk cluster: %d  |  high-risk count: %d / %d",
        high_risk_cluster,
        rfm_df["is_high_risk"].sum(),
        len(rfm_df),
    )
    return rfm_df[["CustomerId", "recency", "frequency", "monetary", "is_high_risk"]]


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (features only, no target)
# ---------------------------------------------------------------------------

def build_feature_pipeline(categorical_cols: list, numerical_cols: list) -> Pipeline:
    """
    Returns a fitted-able sklearn Pipeline that accepts a raw
    customer-level DataFrame and outputs a scaled, encoded array.
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return Pipeline([("preprocessor", preprocessor)])


# ---------------------------------------------------------------------------
# End-to-end processing function
# ---------------------------------------------------------------------------

def load_and_process(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Load raw transactions, engineer features, build RFM proxy label,
    and return a model-ready DataFrame indexed by CustomerId.
    """
    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    # Datetime features on transaction level (pick last per customer)
    dt_transformer = DatetimeFeatures()
    df = dt_transformer.transform(df)

    # Per-customer last transaction time features
    last_tx = (
        df.sort_values("TransactionStartTime")
        .groupby("CustomerId")
        .last()
        .reset_index()
    )[["CustomerId", "transaction_hour", "transaction_day",
       "transaction_month", "transaction_year",
       "ProductCategory", "ChannelId", "ProviderId", "FraudResult"]]

    # Aggregate monetary features
    agg_transformer = AggregateFeatures()
    agg_df = agg_transformer.transform(df)

    # RFM features + proxy label
    rfm_transformer = RFMFeatures()
    rfm_transformer.fit(df)
    rfm_df = rfm_transformer.transform(df)
    rfm_labeled = create_proxy_label(rfm_df)

    # Merge everything
    merged = (
        agg_df
        .merge(last_tx, on="CustomerId", how="left")
        .merge(rfm_labeled, on="CustomerId", how="left")
    )

    logger.info("Feature matrix shape: %s", merged.shape)
    logger.info("High-risk rate: %.2f%%", merged["is_high_risk"].mean() * 100)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed = load_and_process(RAW_PATH)
    processed.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved processed data to %s", PROCESSED_PATH)
