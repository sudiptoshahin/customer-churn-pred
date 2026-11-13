from typing import List, Literal
from pydantic import BaseModel, Field


class Customer(BaseModel):
    tenure: int = Field(..., ge=0, le=100, description="Months with the company")
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_calls: int = Field(..., ge=0)
    senior_citizen: Literal[0, 1]
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    tech_support: Literal["Yes", "No"]
    paperless_billing: Literal["Yes", "No"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 3,
                "monthly_charges": 85.5,
                "total_charges": 256.5,
                "num_support_calls": 4,
                "senior_citizen": 0,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "tech_support": "No",
                "paperless_billing": "Yes",
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]


class BatchRequest(BaseModel):
    customers: List[Customer]


class BatchResponse(BaseModel):
    predictions: List[Prediction]
