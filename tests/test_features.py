import pandas as pd
from src.data.make_dataset import clean_data, generate_synthetic_data
from src.features.build_features import add_engineered_features


def test_synthetic_data_shape():
    df = generate_synthetic_data(200, 1)
    assert len(df) == 200
    assert set(df["churn"].unique()) <= {0, 1}


def test_clean_data_fills_missing():
    df = generate_synthetic_data(100, 1)
    df.loc[0, "monthly_charges"] = None
    out = clean_data(df)
    assert out["monthly_charges"].isna().sum() == 0
    assert "customer_id" not in out.columns


def test_engineered_features():
    df = pd.DataFrame({"tenure": [0, 12], "total_charges": [0.0, 120.0]})
    out = add_engineered_features(df)
    assert out["avg_monthly_spend"].tolist() == [0.0, 10.0]
    assert out["is_new_customer"].tolist() == [1, 0]
