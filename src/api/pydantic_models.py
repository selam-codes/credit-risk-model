"""Request and response schemas for the credit risk API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    total_amount: float = Field(..., description="Sum of all transaction amounts for the customer")
    avg_amount: float = Field(..., description="Average transaction amount")
    transaction_count: int = Field(..., ge=1, description="Total number of transactions")
    std_amount: float = Field(default=0.0, description="Std deviation of transaction amounts")
    recency: int = Field(..., ge=0, description="Days since last transaction")
    frequency: int = Field(..., ge=1, description="Number of transactions (RFM frequency)")
    monetary: float = Field(..., description="Total monetary value of transactions (RFM monetary)")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of last transaction (0-23)")
    transaction_day: int = Field(..., ge=1, le=31, description="Day of month of last transaction")
    transaction_month: int = Field(..., ge=1, le=12, description="Month of last transaction")
    transaction_year: int = Field(..., description="Year of last transaction")
    ProductCategory: str = Field(..., description="Product category of last transaction")
    ChannelId: str = Field(..., description="Channel used for last transaction")
    ProviderId: str = Field(..., description="Provider for last transaction")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of customer being high-risk (0=low risk, 1=high risk)",
    )
    is_high_risk: int = Field(..., description="Binary prediction: 1=high risk, 0=low risk")
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score derived from risk probability (300=worst, 850=best)",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
