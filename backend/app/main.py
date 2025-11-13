from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr

from ml_model import inference
from .dependencies import get_artifacts
from .settings import get_settings


class SummarizeRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]


class SummarizeResponse(BaseModel):
    summary: str


app = FastAPI(title="Transformer Summarizer API")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(
    payload: SummarizeRequest,
    artifacts: inference.ModelArtifacts = Depends(get_artifacts),
):
    settings = get_settings()
    summary = inference.generate_summary(
        payload.text,
        artifacts,
        max_input_length=settings.max_input_length,
        max_output_length=settings.max_output_length,
    )
    return SummarizeResponse(summary=summary)

