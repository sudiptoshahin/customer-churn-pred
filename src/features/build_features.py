"""Feature engineering + sklearn preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns. Used in training AND in the API so they never drift."""
    df = df.copy()
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].clip(lower=1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    return df


ENGINEERED_NUMERIC = ["avg_monthly_spend", "is_new_customer"]


def build_preprocessor(numeric: list, categorical: list) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric + ENGINEERED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
