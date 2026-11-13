"""Create a synthetic telecom-style churn dataset and clean it."""
import numpy as np
import pandas as pd

from src.config import load_config, resolve


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
    )
    tech_support = rng.choice(["Yes", "No"], n_samples, p=[0.3, 0.7])
    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.6, 0.4])
    senior = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    calls = rng.poisson(1.5, n_samples)

    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    monthly = np.round(base + rng.normal(0, 8, n_samples), 2).clip(18, None)
    total = np.round(monthly * tenure + rng.normal(0, 20, n_samples), 2).clip(0, None)

    # Hidden rule that generates churn (so the model has something real to learn)
    logit = (
        -0.8
        + 1.3 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        + 0.6 * (internet == "Fiber optic")
        - 0.03 * tenure
        + 0.015 * (monthly - 60)
        + 0.35 * calls
        + 0.5 * (payment == "Electronic check")
        - 0.6 * (tech_support == "Yes")
        + 0.3 * senior
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n_samples)],
            "tenure": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": calls,
            "senior_citizen": senior,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "tech_support": tech_support,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates/IDs, fix missing values."""
    df = df.drop_duplicates().copy()
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(exclude="number").columns
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def run() -> pd.DataFrame:
    cfg = load_config()
    raw_path = resolve(cfg["paths"]["raw_data"])
    processed_path = resolve(cfg["paths"]["processed_data"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        df = generate_synthetic_data(cfg["data"]["n_samples"], cfg["data"]["random_state"])
        df.to_csv(raw_path, index=False)
        print(f"[data] generated raw data -> {raw_path}")

    clean = clean_data(df)
    clean.to_csv(processed_path, index=False)
    print(f"[data] saved processed data ({len(clean)} rows) -> {processed_path}")
    return clean


if __name__ == "__main__":
    run()
