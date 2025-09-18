"""
Slop web application - Interview transcription management system.
"""

import logging
from typing import TypedDict

import tagflow
from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount

import slop.gemini as gemini
import slop.store as store
from slop import rest, transcribe
from slop.models import Interview, Segment
from slop.parameter import Parameter
from slop.rest import GET, POST, PUT
from slop.store import ModelDecodeError, ModelNotFoundError

logger = logging.getLogger("slop.app")

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

request = rest.request

# ============================================================================
# Type Definitions
# ============================================================================


class AppParams(TypedDict):
    """Application configuration parameters."""

    client: AsyncClient
    google_api_key: str
    gemini_model: str
    interviews_db_path: str
    blobs_db_path: str


# Form validation models using TypedDict for clean **kwargs spreading
class AudioUploadForm(TypedDict):
    audio: UploadFile


class HintForm(TypedDict):
    hint: str | None


class ContentForm(TypedDict):
    content: str


class RenameForm(TypedDict):
    new_name: str


class ContextSegmentsForm(TypedDict):
    context_segments: int


class UpdateSpeakerForm(TypedDict):
    utterance_index: int
    key: str


# ============================================================================
# Application-Specific Context Parameters
# ============================================================================

interview = Parameter[Interview]("app_interview")
segment_index = Parameter[int]("app_segment_index")


# ============================================================================
# Helper Functions
# ============================================================================


def get_segment() -> Segment:
    """Get the current segment from context.

    Raises:
        HTTPException: If segment index is out of bounds.
    """
    interview_obj = interview.get()
    idx = segment_index.get()
    try:
        return interview_obj.segments[idx]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")


# ============================================================================
# Context Loaders
# ============================================================================


def load_interview(interview_id: str) -> Interview:
    """Load an interview by ID or raise HTTP error."""
    try:
        return store.find(Interview, interview_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Interview not found")
    except ModelDecodeError:
        raise HTTPException(status_code=500, detail="Interview data invalid")


def load_segment_index(idx: str) -> int:
    """Load and validate segment index."""
    index = int(idx)
    # Validate segment exists (requires interview context to be set first)
    interview_obj = interview.get()
    if index >= len(interview_obj.segments):
        raise HTTPException(status_code=404, detail="Segment not found")
    return index


# ============================================================================
# Application Middleware
# ============================================================================


class AppParamsMiddleware(BaseHTTPMiddleware):
    """Set up application-wide context like database connections and API keys."""

    def __init__(self, app, params: AppParams | None = None):
        super().__init__(app)
        self.state = params

    async def dispatch(self, req, call_next):
        state: AppParams = self.state

        with (
            gemini.api_key.using(state["google_api_key"]),
            gemini.http_client.using(state["client"]),
            gemini.model.using(state["gemini_model"]),
            store.with_databases(state["interviews_db_path"], state["blobs_db_path"]),
        ):
            return await call_next(req)


# ============================================================================
# Routes
# ============================================================================

routes = [
    GET("/", transcribe.home),
    GET("/home", transcribe.render_home_content),
    GET("/interview-list", transcribe.render_interview_list),
    GET("/audio/{hash_}", transcribe.get_audio),
    POST("/upload", transcribe.upload_audio, form=AudioUploadForm),
    Mount(
        "/interview",
        middleware=[
            Middleware(rest.load(interview, "id", load_interview)),
        ],
        routes=[
            GET("/{id}", transcribe.view_interview),
            GET("/{id}/export", transcribe.export_interview),
            POST("/{id}/transcribe-next", transcribe.transcribe_next_segment),
            POST("/{id}/rename", transcribe.rename_interview, form=RenameForm),
            POST(
                "/{id}/context-segments",
                transcribe.update_context_segments,
                form=ContextSegmentsForm,
            ),
            Mount(
                "/{id}/segment",
                middleware=[
                    Middleware(rest.load(segment_index, "idx", load_segment_index)),
                ],
                routes=[
                    GET("/{idx:int}", transcribe.view_segment),
                    GET("/{idx:int}/edit", transcribe.edit_segment_dialog),
                    POST("/{idx:int}/retranscribe", transcribe.retranscribe_segment),
                    POST(
                        "/{idx:int}/improve-speakers",
                        transcribe.improve_speaker_identification,
                        form=HintForm,
                    ),
                    PUT("/{idx:int}", transcribe.update_segment, form=ContentForm),
                    POST(
                        "/{idx:int}/update-speaker",
                        transcribe.update_speaker,
                        form=UpdateSpeakerForm,
                    ),
                ],
            ),
        ],
    ),
]


# ============================================================================
# Application Factory
# ============================================================================


def create_app(app_state: AppParams) -> Starlette:
    """Create and configure the Starlette application.

    Args:
        app_state: Configuration parameters for the application

    Returns:
        Configured Starlette application
    """
    application = Starlette(routes=routes)

    # Add middleware in reverse order (last added = first executed)
    application.add_middleware(AppParamsMiddleware, params=app_state)
    application.add_middleware(tagflow.DocumentMiddleware)
    application.add_middleware(rest.RequestContextMiddleware)

    return application
