from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reviewguard.api.schemas import AnalyzeRequest, AnalyzeResponse
from reviewguard.config import settings
from reviewguard.ml.inference import ModelNotReadyError, ReviewAnalyzer

app = FastAPI(title="ReviewGuard API", version="0.1.0")
analyzer = ReviewAnalyzer()

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": analyzer.is_ready(),
        "model_name": settings.model_name,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = analyzer.analyze(payload.text)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AnalyzeResponse(**result)

