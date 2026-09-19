"""CustomerIQ v1.1: manifest-backed inference and a static product interface."""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from src.api.manifest import ArtifactManifest
from src.api.observability import MetricsRegistry
from src.api.schemas import CustomerBatch, CustomerPayload, HealthResponse, PredictionResponse
from src.api.security import SecurityMiddleware, SecuritySettings
from src.api.service import ModelService

WEB_DIR = Path(__file__).parent / "web"
logger = logging.getLogger("customeriq")


def create_app(settings: SecuritySettings | None = None, service_factory=ModelService) -> FastAPI:
    settings = settings or SecuritySettings.from_env()
    metrics = MetricsRegistry()

    @asynccontextmanager
    async def lifespan(application):
        application.state.service = service_factory()
        logger.info(json.dumps({"event": "model_loaded", "model_id": application.state.service.manifest.model_id}))
        yield
        application.state.service = None

    application = FastAPI(title="CustomerIQ inference API", version="1.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    application.state.metrics = metrics
    application.add_middleware(SecurityMiddleware, settings=settings, metrics=metrics)
    application.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    if settings.allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"],
                                   expose_headers=["X-Request-ID"], allow_credentials=False)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Pydantic's default errors echo input and can contain NaN / sensitive data.
        errors = error.errors()
        metrics.record_validation(request.url.path, errors)
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in errors]
        return JSONResponse(status_code=422, content={"detail": detail})

    def service(request: Request):
        instance = getattr(request.app.state, "service", None)
        if instance is None:
            raise HTTPException(503, "Model service unavailable.")
        return instance

    @application.get("/", include_in_schema=False)
    def root():
        return FileResponse(WEB_DIR / "index.html")

    @application.get("/docs", include_in_schema=False)
    def api_docs():
        return FileResponse(WEB_DIR / "docs.html")

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request):
        engine = service(request)
        return HealthResponse(
            status="healthy", service="CustomerIQ-Inference-Engine",
            model_loaded=engine.champion_model is not None,
            preprocessor_loaded=engine.preprocessor is not None,
            manifest_available=True, schema_version=engine.manifest.schema_version,
            environment=settings.environment,
        )

    @application.get("/model-info", response_model=ArtifactManifest)
    def model_info(request: Request):
        return service(request).manifest

    @application.get("/input-schema")
    def input_schema():
        return CustomerPayload.model_json_schema()

    @application.get("/metrics", include_in_schema=False)
    def operational_metrics(request: Request):
        engine = service(request)
        return PlainTextResponse(
            metrics.render(engine.manifest.model_id, engine.manifest.schema_version),
            media_type="text/plain; version=0.0.4",
        )

    def infer(customers, request, endpoint):
        started = time.monotonic()
        try:
            responses = service(request).predict_customers(customers)
            metrics.record_inference(endpoint, responses, time.monotonic() - started)
            return responses
        except HTTPException:
            raise
        except Exception as error:
            # Log actionable provenance without payload values or exception text.
            traceback_frames = []
            current = error.__traceback__
            while current is not None:
                traceback_frames.append({
                    "file": Path(current.tb_frame.f_code.co_filename).name,
                    "function": current.tb_frame.f_code.co_name,
                    "line": current.tb_lineno,
                })
                current = current.tb_next
            logger.error(json.dumps({
                "event": "inference_failed",
                "error_type": type(error).__name__,
                "request_id": request.state.request_id,
                "traceback": traceback_frames,
            }))
            raise HTTPException(500, "Prediction failed. Contact support with the response request ID.") from None

    @application.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerPayload, request: Request):
        return infer([customer], request, "/predict")[0]

    @application.post("/predict-batch", response_model=list[PredictionResponse])
    def predict_batch(customers: CustomerBatch, request: Request):
        seen = set()
        duplicates = []
        for i, customer in enumerate(customers):
            if customer.customer_id is not None:
                if customer.customer_id in seen:
                    duplicates.append({"loc": ["body", i, "customer_id"], "msg": "Duplicate customer_id.", "type": "value_error"})
                seen.add(customer.customer_id)
        if duplicates:
            raise HTTPException(422, detail=duplicates)
        return infer(customers, request, "/predict-batch")

    application.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")
    return application


app = create_app()
