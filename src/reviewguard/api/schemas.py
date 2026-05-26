from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class TaskProbability(BaseModel):
    label: str = Field(description="Predicted class label sorted by descending probability.")
    probability: float = Field(ge=0.0, le=1.0, description="Softmax probability for the label.")


class PredictionExplanation(BaseModel):
    sentiment_top_probabilities: list[TaskProbability] = Field(default_factory=list)
    authenticity_top_probabilities: list[TaskProbability] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    token_count: int = Field(ge=0, description="Number of tokens actually scored after truncation.")
    max_length: int = Field(ge=1, description="Maximum token length supported by the exported checkpoint.")
    truncated: bool = Field(description="Whether the original review exceeded the model limit.")


class AnalyzeResponse(BaseModel):
    sentiment_label: str
    sentiment_confidence: float
    authenticity_label: str
    authenticity_confidence: float
    model_name: str
    explanation: PredictionExplanation


class ErrorResponse(BaseModel):
    detail: str
