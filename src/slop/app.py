"""Slop web application - Interview transcription management system."""

import inspect
import logging
import re
from typing import NotRequired, TypedDict

from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from tagflow import TagResponse, document

import slop.gemini as gemini
import slop.store as store
from slop import rest, transcribe
from slop.models import Interview, Segment
from slop.parameter import Parameter
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
    segment_duration_seconds: NotRequired[int]


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


class TranscribeNextForm(TypedDict):
    duration_seconds: NotRequired[int]
    model_name: NotRequired[str]


# ============================================================================
# Application-Specific Context Parameters
# ============================================================================

interview = Parameter[Interview]("app_interview")
segment_index = Parameter[int]("app_segment_index")
segment_duration_seconds = Parameter[int]("app_segment_duration")


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
# Routing Helpers
# ============================================================================


AUDIO_PATH_RE = re.compile(r"^/audio/(?P<hash_>[^/]+)$")
INTERVIEW_PATH_RE = re.compile(
    r"^/interview/(?P<interview_id>[^/]+)(?P<suffix>(?:/.*)?)$"
)
SEGMENT_PATH_RE = re.compile(r"^/segment/(?P<segment_index>\d+)(?P<tail>(?:/.*)?)$")


def set_path_params(req: Request, **params) -> None:
    """Populate Starlette-compatible path params for downstream handlers."""

    req.scope["path_params"] = params


def method_not_allowed(*allowed: str) -> None:
    """Raise a 405 HTTP exception with the provided allowed methods."""

    allow_header = ", ".join(sorted({m.upper() for m in allowed}))
    raise HTTPException(
        status_code=405,
        detail="Method not allowed",
        headers={"allow": allow_header},
    )


async def call_view(handler, *, form_model=None, extra_kwargs=None):
    """Invoke a view function, handling optional form parsing and Tagflow defaults."""

    kwargs = dict(extra_kwargs or {})
    if form_model is not None:
        kwargs.update(await rest.form(form_model))

    if inspect.iscoroutinefunction(handler):
        result = await handler(**kwargs)
    else:
        result = handler(**kwargs)

    return result if result is not None else TagResponse()


async def dispatch_segment(
    req: Request,
    method: str,
    interview_id: str,
    match: re.Match[str],
) -> Response:
    """Route requests that operate on a specific interview segment."""

    tail = match["tail"] or ""
    index = load_segment_index(match["segment_index"])
    set_path_params(req, id=interview_id, idx=index)

    with segment_index.using(index):
        match tail:
            case "":
                if method == "GET":
                    return await call_view(transcribe.view_segment)
                if method == "PUT":
                    return await call_view(
                        transcribe.update_segment, form_model=ContentForm
                    )
                method_not_allowed("GET", "PUT")
            case "/edit":
                if method != "GET":
                    method_not_allowed("GET")
                return await call_view(transcribe.edit_segment_dialog)
            case "/retranscribe":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(transcribe.retranscribe_segment)
            case "/improve-speakers":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(
                    transcribe.improve_speaker_identification,
                    form_model=HintForm,
                )
            case "/update-speaker":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(
                    transcribe.update_speaker, form_model=UpdateSpeakerForm
                )
            case _:
                raise HTTPException(status_code=404, detail="Not found")


async def dispatch_interview(
    req: Request,
    method: str,
    match: re.Match[str],
) -> Response:
    """Route requests for interview-level operations."""

    interview_id = match["interview_id"]
    suffix = match["suffix"] or ""
    interview_obj = load_interview(interview_id)

    with interview.using(interview_obj):
        set_path_params(req, id=interview_id)

        match suffix:
            case "":
                if method != "GET":
                    method_not_allowed("GET")
                return await call_view(transcribe.view_interview)
            case "/export":
                if method != "GET":
                    method_not_allowed("GET")
                return await call_view(transcribe.export_interview)
            case "/transcribe-next":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(
                    transcribe.transcribe_next_segment,
                    form_model=TranscribeNextForm,
                )
            case "/rename":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(
                    transcribe.rename_interview,
                    form_model=RenameForm,
                )
            case "/context-segments":
                if method != "POST":
                    method_not_allowed("POST")
                return await call_view(
                    transcribe.update_context_segments,
                    form_model=ContextSegmentsForm,
                )
            case _:
                if segment_match := SEGMENT_PATH_RE.fullmatch(suffix):
                    return await dispatch_segment(req, method, interview_id, segment_match)
                raise HTTPException(status_code=404, detail="Not found")


async def dispatch_request(req: Request) -> Response:
    """Route the incoming request to the appropriate handler."""

    method = req.method.upper()
    if method == "HEAD":
        method = "GET"

    path = req.url.path
    set_path_params(req)

    match (method, path):
        case ("GET", "/"):
            return await call_view(transcribe.home)
        case ("GET", "/home"):
            return await call_view(transcribe.render_home_content)
        case ("GET", "/interview-list"):
            return await call_view(transcribe.render_interview_list)
        case ("GET", _path) if (match := AUDIO_PATH_RE.fullmatch(path)):
            set_path_params(req, hash_=match["hash_"])
            return await call_view(transcribe.get_audio)
        case ("POST", "/upload"):
            return await call_view(transcribe.upload_audio, form_model=AudioUploadForm)
        case (method, _path) if (match := INTERVIEW_PATH_RE.fullmatch(path)):
            return await dispatch_interview(req, method, match)
        case _:
            raise HTTPException(status_code=404, detail="Not found")


async def handle_request(req: Request) -> Response:
    """Catch-all Starlette handler that injects shared context and dispatches."""

    state: AppParams = req.app.state.app_params
    segment_len = state.get("segment_duration_seconds", 120)

    with (
        rest.request.using(req),
        document(),
        gemini.api_key.using(state["google_api_key"]),
        gemini.http_client.using(state["client"]),
        gemini.uploader.using(gemini.upload),
        gemini.model.using(state["gemini_model"]),
        store.with_databases(state["interviews_db_path"], state["blobs_db_path"]),
        segment_duration_seconds.using(segment_len),
    ):
        response = await dispatch_request(req)
        return response if response is not None else TagResponse()


# ============================================================================
# Application Factory
# ============================================================================


def create_app(app_state: AppParams) -> Starlette:
    """Create and configure the Starlette application."""

    application = Starlette(
        routes=[
            Route(
                "/{path:path}",
                handle_request,
                methods=["GET", "POST", "PUT", "HEAD"],
            )
        ]
    )
    application.state.app_params = app_state
    return application
