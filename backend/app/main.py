"""FastAPI application for FRAUD SHIELD backend.

Provides prediction, model metadata, health check, and UI serving endpoints.
Injects bridge.js into the UI in-memory at response time without modifying disk.
"""

from contextlib import asynccontextmanager
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.explain import generate_reasons
from app.features import extract_features
from app.model_service import model_service
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_shield.api")

# Uptime tracker
SERVER_START_TIME = time.time()

# In-memory rate limiting: ip -> list of timestamps
_rate_limit_records: Dict[str, List[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Simple in-memory sliding window rate limiter."""
    now = time.time()
    window_start = now - 60.0  # 1 minute window
    timestamps = [t for t in _rate_limit_records[client_ip] if t > window_start]
    if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
        return False
    timestamps.append(now)
    _rate_limit_records[client_ip] = timestamps
    return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler to initialize artifacts on startup."""
    logger.info("Initializing FRAUD SHIELD backend services...")
    # Load model and scaler artifacts (fails loudly if missing or incompatible)
    model_service.load_artifacts()
    logger.info(f"Model {model_service.model_name} loaded successfully.")
    yield
    logger.info("Shutting down FRAUD SHIELD backend.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Machine Learning API for Real-Time Financial Fraud Detection",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for bridge.js and any backend assets
static_dir = Path(__file__).resolve().parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Enforce rate limits per client IP."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    # Exempt health checks and documentation
    if not request.url.path.startswith("/api/health"):
        if not check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please retry in a moment."},
            )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return user-friendly 422 error details."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        msg = err.get("msg", "Invalid value")
        errors.append(f"{field}: {msg}" if field else msg)

    logger.info(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors, "message": "Input validation failed."},
    )


@app.get("/bridge.js")
async def get_bridge_js() -> Response:
    """Serve the integration bridge script directly from static directory."""
    bridge_path = static_dir / "bridge.js"
    if bridge_path.exists():
        content = bridge_path.read_text(encoding="utf-8")
        return Response(content=content, media_type="application/javascript")
    return Response(
        content="// bridge.js not yet installed\n",
        media_type="application/javascript",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve frontend/index.html with bridge.js injected in-memory before </body>.

    CRITICAL RULE: The file on disk is never modified.
    """
    html_path = settings.resolved_frontend_html_path
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frontend HTML not found at '{html_path}'.",
        )

    # Read original HTML from disk without modifying it
    html_content = html_path.read_text(encoding="utf-8")

    # Inject bridge.js in-memory right before </body>
    bridge_tag = '<script src="/bridge.js"></script>\n</body>'
    if "</body>" in html_content:
        injected_html = html_content.replace("</body>", bridge_tag, 1)
    else:
        injected_html = html_content + '\n<script src="/bridge.js"></script>'

    return HTMLResponse(content=injected_html, status_code=200)


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health, model status, and uptime."""
    uptime = time.time() - SERVER_START_TIME
    return HealthResponse(
        status="ok" if model_service.is_loaded else "unhealthy",
        model_loaded=model_service.is_loaded,
        model_name=model_service.model_name,
        version=settings.VERSION,
        uptime_seconds=round(uptime, 2),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Retrieve model specifications and benchmark results generated from notebook runs."""
    metrics_path = settings.resolved_metrics_path
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics JSON not found at '{metrics_path}'.",
        )

    try:
        metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse metrics file: {exc}",
        ) from exc

    return ModelInfoResponse(
        model_name=metrics_data.get("model_name", "RandomForestClassifier"),
        display_name=metrics_data.get("display_name", "Random Forest"),
        feature_count=metrics_data.get("feature_count", 9),
        features=metrics_data.get("features", []),
        decision_threshold=settings.DECISION_THRESHOLD,
        production_metrics=metrics_data.get("production_metrics", {}),
        benchmark_comparison=metrics_data.get("benchmark_comparison", []),
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict_fraud(request: PredictRequest) -> PredictResponse:
    """Evaluate transaction inputs against the trained Random Forest model.

    Returns:
        Risk probability, FLAGGED/CLEARED verdict, classification threshold,
        plain-language contributing reasons, model metadata, and latency.
    """
    t0 = time.perf_counter()

    # 1. Feature Engineering
    features_df = extract_features(request)

    # 2. Model Inference
    risk_prob, scaled_features = model_service.predict_risk(features_df)

    # 3. Verdict Determination based on configured threshold
    threshold = settings.DECISION_THRESHOLD
    verdict = "FLAGGED" if risk_prob >= threshold else "CLEARED"

    # 4. Generate plain-language explanation factors
    importances = getattr(model_service.model, "feature_importances_", None)
    reasons = generate_reasons(
        tx_type=request.type,
        amount=request.amount,
        sender_balance=request.sender_balance,
        recipient_balance=request.recipient_balance,
        time_str=request.time,
        fraud_risk=risk_prob,
        feature_importances=importances,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Structured logging (NO PII or sensitive account details)
    logger.info(
        f"Prediction: type={request.type.value} risk={risk_prob:.4f} "
        f"verdict={verdict} latency_ms={elapsed_ms:.2f}"
    )

    return PredictResponse(
        risk=round(risk_prob, 4),
        verdict=verdict,
        threshold=threshold,
        reasons=reasons,
        model=model_service.model_name,
        latency_ms=round(elapsed_ms, 2),
    )
