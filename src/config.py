"""Central config loader. All paths are resolved relative to the project root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> Path:
    """Turn a config path like 'models/x.joblib' into an absolute Path."""
    return ROOT / relative_path
