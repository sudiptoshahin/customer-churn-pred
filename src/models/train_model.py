"""Model factory + training."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor


def get_estimator(model_type: str, params: dict):
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, **params)
    raise ValueError(f"Unknown model type: {model_type}")


def build_pipeline(cfg: dict) -> Pipeline:
    pre = build_preprocessor(cfg["features"]["numeric"], cfg["features"]["categorical"])
    est = get_estimator(cfg["model"]["type"], cfg["model"]["params"])
    return Pipeline([("preprocess", pre), ("model", est)])


def train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline
