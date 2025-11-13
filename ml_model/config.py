from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "mtsamples.csv"
DEFAULT_MODEL_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH: Final[Path] = DEFAULT_MODEL_DIR / "model.pt"
DEFAULT_TOKENIZER_PATH: Final[Path] = DEFAULT_MODEL_DIR / "tokenizer.json"


def ensure_artifact_dirs() -> None:
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

