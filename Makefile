.PHONY: install install-dev data train run test docker-build up down logs retrain clean

install:        ## Install runtime deps into .venv
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

install-dev:    ## Install runtime + dev deps (pytest, jupyter)
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-dev.txt

data:
	.venv/bin/python -m src.data.make_dataset

train:
	.venv/bin/python -m src.pipelines.training_pipeline

run:
	.venv/bin/uvicorn api.main:app --reload --port 8000

test:
	.venv/bin/pytest -v

docker-build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

retrain:
	docker compose run --rm train && curl -X POST http://localhost:8000/reload

clean:
	rm -rf .venv .pytest_cache data/raw/* data/processed/* models/*.joblib models/*.json
