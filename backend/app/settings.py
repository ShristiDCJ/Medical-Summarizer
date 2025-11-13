from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # type: ignore[assignment]

from pydantic import Field


class Settings(BaseSettings):
    model_path: Path = Field(default=Path("artifacts/model.pt"))
    tokenizer_path: Path = Field(default=Path("artifacts/tokenizer.json"))
    max_input_length: int = Field(default=512)
    max_output_length: int = Field(default=128)

    class Config:
        env_prefix = "SUMMARIZER_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

