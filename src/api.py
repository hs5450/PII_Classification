from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import PiiPredictor


app = FastAPI(title="PII Checker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    text_length: int
    pii_risk: bool
    predicted_label: str
    confidence: float | None
    regex_signals: dict[str, bool]
    matched_signals: list[str]
    probabilities: dict[str, float] | None


@lru_cache(maxsize=1)
def get_predictor() -> PiiPredictor:
    return PiiPredictor.from_path()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    try:
        return get_predictor().predict(request.text)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
