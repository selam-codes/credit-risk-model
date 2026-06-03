"""Unit tests for src/data_processing.py."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    AggregateFeatures,
    DatetimeFeatures,
    RFMFeatures,
    WoEEncoder,
    create_proxy_label,
)


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "TransactionId": [f"T{i}" for i in range(10)],
        "CustomerId": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3", "C3", "C4"],
        "Amount": [100, 200, 150, 50, 80, 300, 400, 250, 350, 10],
        "Value": [100, 200, 150, 50, 80, 300, 400, 250, 350, 10],
        "TransactionStartTime": pd.date_range("2024-01-01", periods=10, freq="7D").astype(str),
        "ProductCategory": ["airtime"] * 5 + ["financial_services"] * 5,
        "ChannelId": ["ChannelId_2"] * 10,
        "ProviderId": ["ProviderId_1"] * 10,
        "FraudResult": [0] * 9 + [1],
    })


class TestAggregateFeatures:
    def test_output_columns(self, sample_transactions):
        transformer = AggregateFeatures()
        result = transformer.fit_transform(sample_transactions)
        expected_cols = {"CustomerId", "total_amount", "avg_amount", "transaction_count", "std_amount"}
        assert expected_cols.issubset(set(result.columns))

    def test_customer_count(self, sample_transactions):
        transformer = AggregateFeatures()
        result = transformer.fit_transform(sample_transactions)
        assert len(result) == sample_transactions["CustomerId"].nunique()

    def test_total_amount_is_sum(self, sample_transactions):
        transformer = AggregateFeatures()
        result = transformer.fit_transform(sample_transactions)
        c1_total = result.loc[result["CustomerId"] == "C1", "total_amount"].values[0]
        expected = sample_transactions[sample_transactions["CustomerId"] == "C1"]["Amount"].sum()
        assert c1_total == pytest.approx(expected)

    def test_std_amount_no_nan_for_single_tx(self, sample_transactions):
        transformer = AggregateFeatures()
        result = transformer.fit_transform(sample_transactions)
        # C4 has only 1 transaction — std should be filled with 0, not NaN
        c4_std = result.loc[result["CustomerId"] == "C4", "std_amount"].values[0]
        assert not np.isnan(c4_std)
        assert c4_std == 0.0


class TestDatetimeFeatures:
    def test_output_columns(self, sample_transactions):
        transformer = DatetimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        for col in ["transaction_hour", "transaction_day", "transaction_month", "transaction_year"]:
            assert col in result.columns

    def test_no_row_loss(self, sample_transactions):
        transformer = DatetimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        assert len(result) == len(sample_transactions)

    def test_hour_range(self, sample_transactions):
        transformer = DatetimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        assert result["transaction_hour"].between(0, 23).all()

    def test_month_range(self, sample_transactions):
        transformer = DatetimeFeatures()
        result = transformer.fit_transform(sample_transactions)
        assert result["transaction_month"].between(1, 12).all()


class TestRFMFeatures:
    def test_output_columns(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        result = transformer.transform(sample_transactions)
        assert {"CustomerId", "recency", "frequency", "monetary"}.issubset(result.columns)

    def test_recency_non_negative(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        result = transformer.transform(sample_transactions)
        assert (result["recency"] >= 0).all()

    def test_frequency_equals_transaction_count(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        result = transformer.transform(sample_transactions)
        c1_freq = result.loc[result["CustomerId"] == "C1", "frequency"].values[0]
        assert c1_freq == 3  # C1 has 3 rows

    def test_monetary_equals_amount_sum(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        result = transformer.transform(sample_transactions)
        c1_monetary = result.loc[result["CustomerId"] == "C1", "monetary"].values[0]
        expected = 100 + 200 + 150
        assert c1_monetary == pytest.approx(expected)


class TestCreateProxyLabel:
    def test_binary_label(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        rfm = transformer.transform(sample_transactions)
        labeled = create_proxy_label(rfm, n_clusters=2)
        assert set(labeled["is_high_risk"].unique()).issubset({0, 1})

    def test_output_columns(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        rfm = transformer.transform(sample_transactions)
        labeled = create_proxy_label(rfm, n_clusters=2)
        assert "is_high_risk" in labeled.columns
        assert "CustomerId" in labeled.columns

    def test_no_customer_loss(self, sample_transactions):
        transformer = RFMFeatures()
        transformer.fit(sample_transactions)
        rfm = transformer.transform(sample_transactions)
        labeled = create_proxy_label(rfm, n_clusters=2)
        assert len(labeled) == rfm["CustomerId"].nunique()


class TestWoEEncoder:
    def test_woe_columns_created(self):
        df = pd.DataFrame({
            "ProductCategory": ["airtime", "financial_services", "airtime", "utility_bill"],
            "is_high_risk": [1, 0, 0, 1],
        })
        encoder = WoEEncoder(columns=["ProductCategory"])
        encoder.fit(df)
        result = encoder.transform(df)
        assert "ProductCategory_woe" in result.columns
        assert "ProductCategory" not in result.columns

    def test_original_col_dropped(self):
        df = pd.DataFrame({
            "ChannelId": ["ChannelId_1", "ChannelId_2", "ChannelId_1"],
            "is_high_risk": [1, 0, 1],
        })
        encoder = WoEEncoder(columns=["ChannelId"])
        encoder.fit(df)
        result = encoder.transform(df)
        assert "ChannelId" not in result.columns

    def test_unseen_category_gets_zero(self):
        train = pd.DataFrame({
            "ProductCategory": ["airtime", "financial_services"],
            "is_high_risk": [1, 0],
        })
        test = pd.DataFrame({
            "ProductCategory": ["new_category"],
        })
        encoder = WoEEncoder(columns=["ProductCategory"])
        encoder.fit(train)
        result = encoder.transform(test)
        assert result["ProductCategory_woe"].iloc[0] == 0.0
