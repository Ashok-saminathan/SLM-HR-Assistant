"""FastAPI application.

Run:   SLM_VARIANT=ft_merged uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs

Env vars:
  SLM_CONFIG   path to a model config yaml   (default configs/model.yaml)
  SLM_VARIANT  base | ft_lora | ft_merged | ft_merged_nf4 | ft_merged_int8_cpu   (default ft_merged)
  SLM_LOG      request log path (default experiments/api/requests.jsonl)
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from src.api.service import AssistantService
from src.utils import PROJECT_ROOT, load_config


def _load_real_service() -> tuple[AssistantService, str, str]:
    from src.model.inference import SLMEngine
    from src.optimization.variants import load_variant

    cfg = load_config(os.environ.get("SLM_CONFIG", str(PROJECT_ROOT / "configs/model.yaml")))
    variant = os.environ.get("SLM_VARIANT", "ft_merged")
    model, tok, _ = load_variant(variant, cfg)
    engine = SLMEngine(model, tok, cfg["generation"])
    log = os.environ.get("SLM_LOG", str(PROJECT_ROOT / "experiments/api/requests.jsonl"))
    return AssistantService(engine, log, cfg["retrieval"]["top_k"]), cfg["model"]["name"], variant


def create_app(service: AssistantService | None = None) -> FastAPI:
    """App factory. Pass a service with a fake engine in tests; otherwise the real SLM is loaded at startup."""
    state = {"service": service, "model": "injected", "variant": "injected"} if service else {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if "service" not in state:
            state["service"], state["model"], state["variant"] = _load_real_service()
        yield

    app = FastAPI(title="Nimbus Works HR Policy Assistant (SLM)", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        msgs = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content=ErrorResponse(error="invalid_request", detail=msgs).model_dump())

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = uuid.uuid4().hex[:12]
        return JSONResponse(status_code=500, content=ErrorResponse(request_id=rid, error="internal_error",
                            detail="Unexpected server error. Quote the request_id to the maintainers.").model_dump())

    @app.get("/health", response_model=HealthResponse)
    def health():
        ready = state.get("service") is not None
        return HealthResponse(status="ok" if ready else "loading", model=state.get("model"), variant=state.get("variant"))

    @app.get("/metrics")
    def metrics():
        svc = state.get("service")
        if svc is None:
            raise HTTPException(503, "model not loaded")
        return svc.metrics.snapshot()

    @app.post("/ask", response_model=AskResponse, responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def ask(req: AskRequest):
        svc = state.get("service")
        if svc is None:
            raise HTTPException(503, "model not loaded yet")
        r = svc.ask(req.question)
        return AskResponse(**r.__dict__)

    return app


app = create_app()  # `uvicorn src.api.main:app` - the model is loaded at startup, not at import
