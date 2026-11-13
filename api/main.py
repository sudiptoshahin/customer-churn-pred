"""FastAPI service. Docs at /docs once running."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import BatchRequest, BatchResponse, Customer, Prediction
from src.config import load_config, resolve
from src.models.predict_model import load_model, predict_proba

state = {"model": None}


def _to_prediction(p: float) -> Prediction:
    risk = "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
    return Prediction(churn_probability=round(p, 4), will_churn=p >= 0.5, risk_level=risk)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = resolve(load_config()["paths"]["model"])
    if path.exists():
        state["model"] = load_model(path)
        print(f"[api] model loaded from {path}")
    else:
        print("[api] WARNING: no model found. Run the training pipeline first.")
    yield
    state["model"] = None


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


def _require_model():
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded. Train it first.")
    return state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    model = _require_model()
    p = predict_proba(model, [customer.model_dump()])[0]
    return _to_prediction(p)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    model = _require_model()
    probs = predict_proba(model, [c.model_dump() for c in req.customers])
    return BatchResponse(predictions=[_to_prediction(p) for p in probs])


@app.post("/reload")
def reload_model():
    """Hot-reload the model file after retraining, without restarting."""
    path = resolve(load_config()["paths"]["model"])
    if not path.exists():
        raise HTTPException(404, "Model file not found")
    state["model"] = load_model(path)
    return {"status": "reloaded"}
