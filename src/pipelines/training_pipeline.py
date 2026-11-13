"""End-to-end: data -> features -> train -> evaluate -> save.

Run:  python -m src.pipelines.training_pipeline
"""
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve
from src.data.make_dataset import run as prepare_data
from src.evaluation.evaluate import evaluate
from src.features.build_features import add_engineered_features
from src.models.train_model import build_pipeline, train


def run() -> dict:
    cfg = load_config()
    df = prepare_data()
    df = add_engineered_features(df)

    target = cfg["data"]["target"]
    X, y = df.drop(columns=[target]), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    model = train(build_pipeline(cfg), X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    model_path = resolve(cfg["paths"]["model"])
    metrics_path = resolve(cfg["paths"]["metrics"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train] metrics: {metrics}")
    print(f"[train] model saved -> {model_path}")
    return metrics


if __name__ == "__main__":
    run()
