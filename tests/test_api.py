SAMPLE = {
    "tenure": 3, "monthly_charges": 85.5, "total_charges": 256.5,
    "num_support_calls": 4, "senior_citizen": 0,
    "contract": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "tech_support": "No",
    "paperless_billing": "Yes",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"customers": [SAMPLE, SAMPLE]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_validation_error(client):
    bad = {**SAMPLE, "contract": "Weekly"}
    assert client.post("/predict", json=bad).status_code == 422
