from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from ml_model import inference
from .settings import Settings, get_settings


@lru_cache
def _load_artifacts(settings: Settings) -> inference.ModelArtifacts:
    return inference.load_artifacts(
        model_path=settings.model_path,
        tokenizer_path=settings.tokenizer_path,
    )


def get_artifacts(
    settings: Annotated[Settings, Depends(get_settings)]
) -> inference.ModelArtifacts:
    return _load_artifacts(settings)

