"""Slop web application - Tape transcription management system."""

import inspect
import logging
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
import slop.rest as rest
import slop.store as store
import slop.transcribe as views
from slop.models import (
    Part,
    Tape,
)
from slop.models import (
    get_part as load_part_model,
)
from slop.models import (
    get_tape as load_tape_model,
)
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
part = Parameter[Part]("app_part")
part_index = Parameter[int]("app_part_index")
part_duration_seconds = Parameter[int]("app_part_duration")


# ============================================================================
# Helper Functions
# ============================================================================


def get_part() -> Part:
    """Return the current part from context."""

    try:
        return part.get()
    except LookupError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=404, detail="Part not found") from exc


# ============================================================================
# Context Loaders
# ============================================================================


def load_tape(tape_id: str) -> Tape:
    """Load an tape by ID or raise HTTP error."""
    try:
        return load_tape_model(tape_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Tape not found")
    except ModelDecodeError:
        raise HTTPException(status_code=500, detail="Tape data invalid")


def load_part(part_id: str) -> Part:
    """Load a part or raise an HTTP error."""

    try:
        return load_part_model(part_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail="Part not found")
    except ModelDecodeError:
        raise HTTPException(status_code=500, detail="Part data invalid")


# ============================================================================
# Routing Helpers
# ============================================================================


def set_path_params(req: Request, **params) -> None:
    """Populate Starlette-compatible path params for downstream handlers."""

    req.scope["path_params"] = params


def allow_methods(*allowed: str) -> None:
    """Raise a 405 HTTP exception with the provided allowed methods."""
    if request.get().method in allowed:
        return

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


def normalize_method(method: str) -> str:
    method = method.upper()
    return "GET" if method == "HEAD" else method


def split_path(path: str) -> tuple[str, ...]:
    if path in {"", "/"}:
        return ()
    return tuple(segment for segment in path.strip("/").split("/") if segment)


async def resolve_route(req: Request):
    method = normalize_method(req.method)
    segments = split_path(req.url.path)

    match segments:
        case ():
            allow_methods("GET")
            return views.home()

        case ("tapes",):
            allow_methods("GET", "POST")
            match method:
                case "GET":
                    return views.list_tapes_view()
                case "POST":
                    return await views.create_tape(**(await rest.form(AudioUploadForm)))

        case ("tapes", tape_id, *tail):
            with tape.using(load_tape(tape_id)):
                match tail:
                    case ():
                        allow_methods("GET", "PATCH")
                        match method:
                            case "GET":
                                return views.view_tape()
                            case "PATCH":
                                return await views.patch_tape()

                    case ("parts",):
                        allow_methods("GET")
                        return views.list_parts_view()

                    case ("jobs",):
                        allow_methods("POST")
                        return await views.tape_jobs()

                    case ("export",):
                        allow_methods("GET")
                        return views.export_tape()

                    case ("media",):
                        allow_methods("GET")
                        return views.get_tape_media()

        case ("parts", part_id, *tail):
            with part.using(load_part(part_id)):
                match tail:
                    case ():
                        allow_methods("GET", "POST")
                        match method:
                            case "GET":
                                return views.view_part()
                            case "POST":
                                return await views.update_part(
                                    **(await rest.form(ContentForm))
                                )

                    case ("edit",):
                        allow_methods("GET")
                        return views.edit_part_dialog()

                    case ("jobs",):
                        allow_methods("POST")
                        raise NotImplementedError("Part job creation not implemented")

        case ("audio", blob):
            allow_methods("GET")
            return views.get_audio(blob)

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
        match await resolve_route(req):
            case Response() as response:
                return response
            case _:
                return TagResponse()


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
