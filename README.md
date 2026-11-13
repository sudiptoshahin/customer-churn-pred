# Customer Churn Prediction

FastAPI service + scikit-learn pipeline that predicts whether a telecom customer will churn.
Data is synthetic (generated automatically), so the project runs with zero downloads.

## Quick start (Docker – recommended)

```bash
docker compose up -d --build     # build + start (auto-trains on first run)
docker compose logs -f api       # watch logs
docker compose down              # stop
```
Open http://localhost:8000/docs

## Local Python install

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m src.pipelines.training_pipeline      # train
uvicorn api.main:app --reload --port 8000      # serve
pytest -v                                      # test
jupyter lab                                    # notebooks
```

## Try the API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure": 3, "monthly_charges": 85.5, "total_charges": 256.5,
  "num_support_calls": 4, "senior_citizen": 0, "contract": "Month-to-month",
  "internet_service": "Fiber optic", "payment_method": "Electronic check",
  "tech_support": "No", "paperless_billing": "Yes"}'
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness + model-loaded check |
| POST | /predict | Score one customer |
| POST | /predict/batch | Score many customers |
| POST | /reload | Load a newly trained model without restart |

## Layout
| Path | Purpose |
|---|---|
| `src/data` | Generate + clean data |
| `src/features` | Feature engineering, preprocessing |
| `src/models` | Model factory, training, prediction |
| `src/evaluation` | Metrics |
| `src/pipelines` | End-to-end training pipeline |
| `api/` | FastAPI app + Pydantic schemas |
| `configs/config.yaml` | Paths, features, model hyper-parameters |
| `notebooks/` | Exploration and experiments |
| `tests/` | pytest suite |
