"""
Low-latency REST API serving the registered model.

CIM's posting: "Transition models from notebook prototypes into scalable batch
jobs or low-latency REST APIs." This is that transition -- the model is pulled
from the MLflow Model Registry by ALIAS (not a hardcoded path), so promoting a
new champion swaps the served model without touching this code.

Endpoints:
  GET  /health   -- liveness + which model version is loaded
  POST /predict  -- single loan, returns probability + risk band
  POST /batch    -- many loans at once
  GET  /metrics  -- request count and latency percentiles (ops visibility)
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "cre-default-risk")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request."""
    try:
        load_model()
        print(f"Loaded {MODEL_NAME}@{MODEL_ALIAS} (v{_model_version})")
    except Exception as exc:
        print(f"[warn] model not loaded at startup: {exc}")
    yield


app = FastAPI(
    title="CRE Default Risk API",
    description="Serves the registered cre-default-risk model from the MLflow registry.",
    version="1.0.0",
    lifespan=lifespan,
)

_model = None
_model_version = None
_latencies: list[float] = []


class Loan(BaseModel):
    """The serving contract. Pydantic enforces it at the edge -- this is how you
    stop feature skew before it reaches the model."""
    loan_amount: float = Field(..., gt=0, examples=[6_000_000])
    ltv: float = Field(..., ge=0, le=1.5, examples=[0.68])
    dscr: float = Field(..., gt=0, examples=[1.35])
    occupancy_rate: float = Field(..., ge=0, le=1, examples=[0.91])
    interest_rate: float = Field(..., gt=0, examples=[5.8])
    year_built: int = Field(..., ge=1800, le=2100, examples=[1998])
    loan_term_months: int = Field(..., gt=0, examples=[120])
    cap_rate: float = Field(..., gt=0, le=1, examples=[0.058])
    market_vacancy: float = Field(..., ge=0, le=1, examples=[0.07])
    sponsor_experience_years: float = Field(..., ge=0, examples=[12.0])
    noi: float = Field(..., examples=[350_000])
    property_type: Literal["office", "retail", "industrial", "multifamily"]
    region: Literal["west", "southwest", "midwest", "northeast", "southeast"]


class Prediction(BaseModel):
    default_probability: float
    risk_band: str
    model_name: str
    model_version: str
    latency_ms: float


def load_model():
    """Load by alias so promotion in the registry changes what we serve."""
    global _model, _model_version
    if _model is None:
        uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        _model = mlflow.sklearn.load_model(uri)
        try:
            from mlflow.tracking import MlflowClient
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            _model_version = str(mv.version)
        except Exception:
            _model_version = "unknown"
    return _model


def risk_band(p: float) -> str:
    """Translate a probability into something a credit committee can act on."""
    if p < 0.05:
        return "low"
    if p < 0.15:
        return "moderate"
    if p < 0.30:
        return "elevated"
    return "high"


@app.get("/health")
def health() -> dict:
    ok = _model is not None
    return {
        "status": "healthy" if ok else "model_not_loaded",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": _model_version,
    }


@app.post("/predict", response_model=Prediction)
def predict(loan: Loan) -> Prediction:
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    t0 = time.perf_counter()
    df = pd.DataFrame([loan.model_dump()])
    prob = float(model.predict_proba(df)[:, 1][0])
    ms = (time.perf_counter() - t0) * 1000
    _latencies.append(ms)
    return Prediction(
        default_probability=round(prob, 6),
        risk_band=risk_band(prob),
        model_name=MODEL_NAME,
        model_version=_model_version or "unknown",
        latency_ms=round(ms, 2),
    )


@app.post("/batch")
def batch(loans: list[Loan]) -> dict:
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    t0 = time.perf_counter()
    df = pd.DataFrame([l.model_dump() for l in loans])
    probs = model.predict_proba(df)[:, 1]
    ms = (time.perf_counter() - t0) * 1000
    _latencies.append(ms)
    return {
        "n": len(loans),
        "predictions": [
            {"default_probability": round(float(p), 6), "risk_band": risk_band(float(p))}
            for p in probs
        ],
        "total_latency_ms": round(ms, 2),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Latency degradation is one of the four things we monitor."""
    if not _latencies:
        return {"requests": 0}
    arr = np.array(_latencies)
    return {
        "requests": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }
