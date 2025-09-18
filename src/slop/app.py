import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import tagflow
from fastapi import FastAPI
from httpx import AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

import slop.gemini as gemini
import slop.models as models
import slop.transcribe

logger = logging.getLogger("slop.app")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class AppState:
    client: AsyncClient
    google_api_key: str
    gemini_model: str
    interviews_db_path: str
    blobs_db_path: str


def configure_app(application: FastAPI):
    """Create and configure the FastAPI application."""

    class AppStateMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            state: AppState = application.state.state
            with (
                gemini.api_key.using(state.google_api_key),
                gemini.http_client.using(state.client),
                gemini.model.using(state.gemini_model),
                models.with_databases(state.interviews_db_path, state.blobs_db_path),
            ):
                return await call_next(request)

    application.add_middleware(AppStateMiddleware)
    application.add_middleware(tagflow.DocumentMiddleware)
    application.include_router(slop.transcribe.app)

    return application


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ["GOOGLE_API_KEY"]
    gemini_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    data_dir = Path(os.environ["IEVA_DATA"])

    data_dir.mkdir(parents=True, exist_ok=True)

    interviews_db_path = str(data_dir / "interviews.db")
    blobs_db_path = str(data_dir / "blobs.db")

    with models.with_databases(interviews_db_path, blobs_db_path):
        models.init_databases()

    async with AsyncClient() as client:
        app.state.state = AppState(
            client=client,
            google_api_key=api_key,
            gemini_model=gemini_model,
            interviews_db_path=interviews_db_path,
            blobs_db_path=blobs_db_path,
        )
        logger.info("App state initialized")
        yield
        logger.info("App state cleanup complete")


app = configure_app(
    FastAPI(
        title="Ieva's Interviews",
        lifespan=lifespan,
    )
)
