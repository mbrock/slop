"""Slop web application - Tape transcription management system."""

import inspect
import logging
from typing import NoReturn, NotRequired, TypedDict

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
from slop.models import Tape, Part
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
    tapes_db_path: str
    blobs_db_path: str
    part_duration_seconds: NotRequired[int]


# Form validation models using TypedDict for clean **kwargs spreading
class AudioUploadForm(TypedDict):
    audio: UploadFile

class ContentForm(TypedDict):
    content: str


# ============================================================================
# Application-Specific Context Parameters
# ============================================================================

tape = Parameter[Tape]("app_tape")
part_index = Parameter[int]("app_part_index")
part_duration_seconds = Parameter[int]("app_part_duration")


# ============================================================================
# Helper Functions
# ============================================================================


def get_part() -> Part:
    """Get the current part from context.

    Raises:
        HTTPException: If part index is out of bounds.
    """
    tape_obj = tape.get()
    idx = part_index.get()
    try:
        return tape_obj.parts[idx]
    except IndexError:
        raise HTTPException(status_code=404, detail="Part not found")


# ============================================================================
# Context Loaders
# ============================================================================


def load_tape(tape_id: str) -> Tape:
    """Load an tape by ID or raise HTTP error."""
    try:
        return store.find(Tape, tape_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Tape not found")
    except ModelDecodeError:
        raise HTTPException(status_code=500, detail="Tape data invalid")


def load_part_index(idx: str) -> int:
    """Load and validate part index."""
    index = int(idx)
    # Validate part exists (requires tape context to be set first)
    tape_obj = tape.get()
    if index >= len(tape_obj.parts):
        raise HTTPException(status_code=404, detail="Part not found")
    return index


# ============================================================================
# Routing Helpers
# ============================================================================


def parse_path(path: str) -> tuple[str, ...] | None:
    """Split a request path into meaningful parts.

    Returns ``None`` when the path contains unsupported constructs such as
    trailing slashes (other than ``/`` itself) or empty parts, to mirror the
    404 behaviour of the previous regex-based routing.
    """

    if path == "/":
        return ()

    if path.endswith("/"):
        return None

    stripped = path.lstrip("/")
    if not stripped:
        return ()

    parts = tuple(stripped.split("/"))
    return None if any(part == "" for part in parts) else parts


def set_path_params(req: Request, **params) -> None:
    """Populate Starlette-compatible path params for downstream handlers."""

    req.scope["path_params"] = params


def method_not_allowed(*allowed: str) -> NoReturn:
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


async def dispatch_part(
    req: Request,
    method: str,
    tape_id: str,
    part_idx: str,
    tail: tuple[str, ...],
) -> Response:
    """Route requests that operate on a specific tape part."""

    if not part_idx.isdigit():
        raise HTTPException(status_code=404, detail="Not found")

    index = load_part_index(part_idx)
    set_path_params(req, id=tape_id, idx=index)

    with part_index.using(index):
        match (method, tail):
            case ("GET", ()):
                return await call_view(transcribe.view_part)
            case ("PATCH", ()):
                return await call_view(
                    transcribe.update_part,
                    form_model=ContentForm,
                )
            case ("GET", ("edit",)):
                return await call_view(transcribe.edit_part_dialog)
            case ("POST", ("jobs",)):
                return await call_view(transcribe.part_jobs)
            case (_, ()):
                method_not_allowed("GET", "PATCH")
            case (_, ("edit",)):
                method_not_allowed("GET")
            case _:
                raise HTTPException(status_code=404, detail="Not found")


async def dispatch_tape(
    req: Request,
    method: str,
    tape_id: str,
    suffix: tuple[str, ...],
) -> Response:
    """Route requests for tape-level operations."""

    tape_obj = load_tape(tape_id)

    with tape.using(tape_obj):
        set_path_params(req, id=tape_id)

        match (method, suffix):
            case ("GET", ()):
                return await call_view(transcribe.view_tape)
            case ("PATCH", ()):
                return await call_view(transcribe.patch_tape)
            case ("GET", ("export",)):
                return await call_view(transcribe.export_tape)
            case ("GET", ("media",)):
                return await call_view(transcribe.get_tape_media)
            case ("POST", ("jobs",)):
                return await call_view(transcribe.tape_jobs)
            case ("GET", ("parts",)):
                return await call_view(transcribe.list_parts_view)
            case (method, ("parts", part_idx, *part_tail)) if part_idx.isdigit():
                return await dispatch_part(
                    req,
                    method,
                    tape_id,
                    part_idx,
                    tuple(part_tail),
                )
            case (_, ()):
                method_not_allowed("GET", "PATCH")
            case (_, ("export",)):
                method_not_allowed("GET")
            case (_, ("media",)):
                method_not_allowed("GET")
            case (_, ("jobs",)):
                method_not_allowed("POST")
            case (_, ("parts",)):
                method_not_allowed("GET")
            case (_, ("parts", _, *_)):
                raise HTTPException(status_code=404, detail="Not found")
            case _:
                raise HTTPException(status_code=404, detail="Not found")


async def dispatch_request(req: Request) -> Response:
    """Route the incoming request to the appropriate handler."""

    method = req.method.upper()
    if method == "HEAD":
        method = "GET"

    path = req.url.path
    set_path_params(req)

    parts = parse_path(path)
    if parts is None:
        raise HTTPException(status_code=404, detail="Not found")

    match (method, parts):
        case ("GET", ()):
            return await call_view(transcribe.home)
        case ("GET", ("tapes",)):
            partial = req.query_params.get("partial") == "list"
            return await call_view(
                transcribe.list_tapes_view,
                extra_kwargs={"partial": partial},
            )
        case ("POST", ("tapes",)):
            return await call_view(transcribe.create_tape, form_model=AudioUploadForm)
        case ("GET", ("audio", hash_)):
            set_path_params(req, hash_=hash_)
            return await call_view(transcribe.get_audio)
        case (method, ("tapes", tape_id, *suffix)):
            return await dispatch_tape(
                req,
                method,
                tape_id,
                tuple(suffix),
            )
        case _:
            raise HTTPException(status_code=404, detail="Not found")


async def handle_request(req: Request) -> Response:
    """Catch-all Starlette handler that injects shared context and dispatches."""

    state: AppParams = req.app.state.app_params
    part_len = state.get("part_duration_seconds", 120)

    with (
        rest.request.using(req),
        document(),
        gemini.api_key.using(state["google_api_key"]),
        gemini.http_client.using(state["client"]),
        gemini.uploader.using(gemini.upload),
        gemini.model.using(state["gemini_model"]),
        store.with_databases(state["tapes_db_path"], state["blobs_db_path"]),
        part_duration_seconds.using(part_len),
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
                methods=["GET", "POST", "PUT", "PATCH", "HEAD"],
            )
        ]
    )
    application.state.app_params = app_state
    return application
