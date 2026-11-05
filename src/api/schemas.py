"""Pydantic request / response schemas for the HR policy assistant API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(..., description="Employee question about HR policy", examples=["How many sick days do I get?"])

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        v = " ".join(v.split())  # collapse whitespace / newlines
        if len(v) < 5:
            raise ValueError("question must be at least 5 characters")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question must be at most {MAX_QUESTION_CHARS} characters")
        if not any(ch.isalpha() for ch in v):
            raise ValueError("question must contain letters")
        if any(ord(ch) < 32 for ch in v):
            raise ValueError("question contains control characters")
        return v


class AskResponse(BaseModel):
    request_id: str
    answer: str
    answerable: bool
    source: str | None = Field(None, description="Policy id the answer is based on")
    retrieved_policy_id: str = Field(..., description="Policy excerpt that was given to the model")
    degraded: bool = Field(False, description="True when the model output failed validation and a safe fallback was returned")
    latency_ms: float


class ErrorResponse(BaseModel):
    request_id: str | None = None
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: str | None
    variant: str | None
