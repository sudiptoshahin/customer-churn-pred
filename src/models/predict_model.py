"""Load a trained pipeline and score customers."""
import joblib
import pandas as pd

from src.features.build_features import add_engineered_features


def load_model(path):
    return joblib.load(path)


def predict_proba(model, records: list) -> list:
    df = add_engineered_features(pd.DataFrame(records))
    return model.predict_proba(df)[:, 1].tolist()
