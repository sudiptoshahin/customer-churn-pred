import pytest
from fastapi.testclient import TestClient

from src.config import load_config, resolve
from src.pipelines import training_pipeline


@pytest.fixture(scope="session")
def client():
    if not resolve(load_config()["paths"]["model"]).exists():
        training_pipeline.run()
    from api.main import app
    with TestClient(app) as c:
        yield c
