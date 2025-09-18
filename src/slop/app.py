"""Slop web application - Tape transcription management system."""

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from re import Match
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
from slop.models import (
    Part,
    Tape,
    get_part as load_part_model,
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


RouteHandler = Callable[[Request, Match[str]], Awaitable[Response]]


@dataclass(frozen=True)
class RouteDef:
    """Single regex-based route definition."""

    method: str
    pattern: re.Pattern[str]
    handler: RouteHandler


async def call_with_tape(
    req: Request,
    tape_id: str,
    view,
    *,
    form_model=None,
    extra_kwargs=None,
) -> Response:
    """Load tape context and invoke the given view."""

    tape_obj = load_tape(tape_id)
    with tape.using(tape_obj):
        set_path_params(req, id=tape_id)
        return await call_view(
            view,
            form_model=form_model,
            extra_kwargs=extra_kwargs,
        )


async def call_with_part(
    req: Request,
    part_id: str,
    view,
    *,
    form_model=None,
    extra_kwargs=None,
) -> Response:
    """Load part context before invoking the view."""

    part_obj = load_part(part_id)
    tape_obj = load_tape(part_obj.tape_id)

    try:
        index = tape_obj.part_ids.index(part_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Part not found for tape")

    with tape.using(tape_obj), part.using(part_obj), part_index.using(index):
        set_path_params(req, id=part_id, tape_id=tape_obj.id, idx=index)
        return await call_view(
            view,
            form_model=form_model,
            extra_kwargs=extra_kwargs,
        )


async def route_home(req: Request, match: Match[str]) -> Response:
    set_path_params(req)
    return await call_view(transcribe.home)


async def route_list_tapes(req: Request, match: Match[str]) -> Response:
    set_path_params(req)
    partial = req.query_params.get("partial") == "list"
    return await call_view(
        transcribe.list_tapes_view,
        extra_kwargs={"partial": partial},
    )


async def route_create_tape(req: Request, match: Match[str]) -> Response:
    set_path_params(req)
    return await call_view(
        transcribe.create_tape,
        form_model=AudioUploadForm,
    )


async def route_get_audio(req: Request, match: Match[str]) -> Response:
    hash_ = match.group("hash_")
    set_path_params(req, hash_=hash_)
    return await call_view(transcribe.get_audio)


async def route_get_tape(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.view_tape)


async def route_patch_tape(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.patch_tape)


async def route_export_tape(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.export_tape)


async def route_media_tape(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.get_tape_media)


async def route_tape_jobs(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.tape_jobs)


async def route_list_parts(req: Request, match: Match[str]) -> Response:
    return await call_with_tape(req, match.group("tape_id"), transcribe.list_parts_view)


async def route_get_part(req: Request, match: Match[str]) -> Response:
    return await call_with_part(
        req,
        match.group("part_id"),
        transcribe.view_part,
    )


async def route_patch_part(req: Request, match: Match[str]) -> Response:
    return await call_with_part(
        req,
        match.group("part_id"),
        transcribe.update_part,
        form_model=ContentForm,
    )


async def route_edit_part(req: Request, match: Match[str]) -> Response:
    return await call_with_part(
        req,
        match.group("part_id"),
        transcribe.edit_part_dialog,
    )


async def route_part_jobs(req: Request, match: Match[str]) -> Response:
    return await call_with_part(
        req,
        match.group("part_id"),
        transcribe.part_jobs,
    )


ROUTES: tuple[RouteDef, ...] = (
    RouteDef("GET", re.compile(r"^/$"), route_home),
    RouteDef("GET", re.compile(r"^/tapes$"), route_list_tapes),
    RouteDef("POST", re.compile(r"^/tapes$"), route_create_tape),
    RouteDef("GET", re.compile(r"^/audio/(?P<hash_>[^/]+)$"), route_get_audio),
    RouteDef("GET", re.compile(r"^/tapes/(?P<tape_id>[^/]+)$"), route_get_tape),
    RouteDef("PATCH", re.compile(r"^/tapes/(?P<tape_id>[^/]+)$"), route_patch_tape),
    RouteDef("GET", re.compile(r"^/tapes/(?P<tape_id>[^/]+)/export$"), route_export_tape),
    RouteDef("GET", re.compile(r"^/tapes/(?P<tape_id>[^/]+)/media$"), route_media_tape),
    RouteDef("POST", re.compile(r"^/tapes/(?P<tape_id>[^/]+)/jobs$"), route_tape_jobs),
    RouteDef("GET", re.compile(r"^/tapes/(?P<tape_id>[^/]+)/parts$"), route_list_parts),
    RouteDef(
        "GET",
        re.compile(r"^/parts/(?P<part_id>[^/]+)$"),
        route_get_part,
    ),
    RouteDef(
        "PATCH",
        re.compile(r"^/parts/(?P<part_id>[^/]+)$"),
        route_patch_part,
    ),
    RouteDef(
        "GET",
        re.compile(r"^/parts/(?P<part_id>[^/]+)/edit$"),
        route_edit_part,
    ),
    RouteDef(
        "POST",
        re.compile(r"^/parts/(?P<part_id>[^/]+)/jobs$"),
        route_part_jobs,
    ),
)


async def dispatch_request(req: Request) -> Response:
    """Route the incoming request to the appropriate handler."""

    method = req.method.upper()
    if method == "HEAD":
        method = "GET"

    path = req.url.path
    set_path_params(req)

    matches: list[tuple[RouteDef, Match[str]]] = []
    allowed_methods: set[str] = set()

    for route in ROUTES:
        match = route.pattern.fullmatch(path)
        if match is None:
            continue
        matches.append((route, match))
        allowed_methods.add(route.method)

    for route, match in matches:
        if route.method == method:
            return await route.handler(req, match)

    if allowed_methods:
        method_not_allowed(*allowed_methods)

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
