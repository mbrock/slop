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
    data_dir: Path


def configure_app(application: FastAPI):
    """Create and configure the FastAPI application."""

    class AppStateMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            state = application.state.state
            with gemini.api_key.using(state.google_api_key):
                with gemini.http_client.using(state.client):
                    with gemini.model.using(state.gemini_model):
                        # Create per-request database connections
                        with models.sqlite(state.data_dir, "interviews.db") as interviews_conn:
                            with models.interviews_db.using(interviews_conn):
                                with models.sqlite(state.data_dir, "blobs.db") as blobs_conn:
                                    with models.blobs_db.using(blobs_conn):
                                        response = await call_next(request)
                                        return response

    application.add_middleware(AppStateMiddleware)
    application.add_middleware(tagflow.DocumentMiddleware)
    application.include_router(slop.transcribe.app)

    return application


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable must be set")
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not gemini_model:
        raise RuntimeError("GEMINI_MODEL environment variable must be set")
    data_dir_env = os.getenv("IEVA_DATA")
    if not data_dir_env:
        raise RuntimeError("IEVA_DATA environment variable must be set")
    data_dir = Path(data_dir_env)
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize databases (creates tables if needed)
    with models.sqlite(data_dir, "interviews.db") as conn:
        with models.interviews_db.using(conn):
            with models.sqlite(data_dir, "blobs.db") as blobs_conn:
                with models.blobs_db.using(blobs_conn):
                    models.init_databases()
    
    async with AsyncClient() as client:
        app.state.state = AppState(
            client=client,
            google_api_key=api_key,
            gemini_model=gemini_model,
            data_dir=data_dir,
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
