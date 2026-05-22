from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class AnalyzeResponse(BaseModel):
    sentiment_label: str
    sentiment_confidence: float
    authenticity_label: str
    authenticity_confidence: float
    model_name: str


class ErrorResponse(BaseModel):
    detail: str

