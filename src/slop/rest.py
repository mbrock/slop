"""Lightweight request helpers shared across the application."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Awaitable, Callable, Sequence, Type, TypeVar

from pydantic import ConfigDict, TypeAdapter, ValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request

from slop.parameter import Parameter

T = TypeVar("T")
R = TypeVar("R")

# ============================================================================
# Context Parameters
# ============================================================================

request = Parameter[Request]("rest_request")


# ============================================================================
# Validation helpers
# ============================================================================


def _ensure_arbitrary_types(model: Type[Any]) -> None:
    config = getattr(model, "__pydantic_config__", None)
    if not getattr(config, "arbitrary_types_allowed", False):
        try:
            setattr(
                model, "__pydantic_config__", ConfigDict(arbitrary_types_allowed=True)
            )
        except (AttributeError, TypeError):
            pass


def _adapter(model: Type[T]) -> TypeAdapter[T]:
    _ensure_arbitrary_types(model)
    return TypeAdapter(model)


def _format_validation_error(exc: ValidationError) -> str:
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")
    return "; ".join(errors)


def _validate(model: Type[T], payload: Any) -> T:
    try:
        return _adapter(model).validate_python(payload)
    except ValidationError as exc:
        detail = _format_validation_error(exc)
        raise HTTPException(
            status_code=400, detail=f"Validation error: {detail}"
        ) from exc


async def parse_body(model: Type[T], *, source: str = "form") -> T:
    """Validate the incoming request body with minimal ceremony."""

    req = request.get()
    if source == "json":
        payload = await req.json()
    elif source == "form":
        payload = await req.form()
    else:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported body source '{source}'")

    return _validate(model, payload)


async def form[T](model: Type[T]) -> T:
    """Maintain the legacy helper for HTML forms."""

    return await parse_body(model, source="form")


# ============================================================================
# Endpoint decorators
# ============================================================================


Endpoint = Callable[..., Awaitable[R] | R]


def endpoint(
    func: Endpoint | None = None,
    *,
    body: Type[Any] | None = None,
    source: str = "form",
    into: str = "payload",
    default_response: Callable[[], Any] | None = None,
) -> Endpoint:
    """Wrap a handler so it can opt into body validation without fuss."""

    def decorator(handler: Endpoint) -> Endpoint:
        @wraps(handler)
        async def wrapper(*args: object, **kwargs: object):
            if body is not None and into not in kwargs:
                kwargs[into] = await parse_body(body, source=source)

            result = handler(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result

            if result is None and default_response is not None:
                return default_response()

            return result

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)

    return decorator


def html(
    func: Endpoint | None = None,
    *,
    body: Type[Any] | None = None,
    source: str = "form",
    into: str = "payload",
) -> Endpoint:
    from tagflow import TagResponse

    return endpoint(
        func, body=body, source=source, into=into, default_response=TagResponse
    )


# ============================================================================
# Route descriptors for manual dispatch tables
# ============================================================================


RouteSpec = tuple[tuple[str, ...], str, Endpoint, str | None]


def route(
    methods: Sequence[str],
    path: str,
    handler: Endpoint,
    *,
    name: str | None = None,
) -> RouteSpec:
    return (tuple(method.upper() for method in methods), path, handler, name)


def GET(path: str, handler: Endpoint, *, name: str | None = None) -> RouteSpec:
    return route(["GET"], path, handler, name=name)


def POST(path: str, handler: Endpoint, *, name: str | None = None) -> RouteSpec:
    return route(["POST"], path, handler, name=name)


def PUT(path: str, handler: Endpoint, *, name: str | None = None) -> RouteSpec:
    return route(["PUT"], path, handler, name=name)


def PATCH(path: str, handler: Endpoint, *, name: str | None = None) -> RouteSpec:
    return route(["PATCH"], path, handler, name=name)


def DELETE(path: str, handler: Endpoint, *, name: str | None = None) -> RouteSpec:
    return route(["DELETE"], path, handler, name=name)


# ============================================================================
# Simple bind helpers for request metadata
# ============================================================================


def bind_path(name: str) -> str:
    try:
        return request.get().path_params[name]
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"Missing path parameter '{name}'"
        ) from exc


def bind_query(name: str, default: Any | None = None) -> Any:
    value = request.get().query_params.get(name, default)
    return value


def bind_header(name: str, default: Any | None = None) -> Any:
    return request.get().headers.get(name, default)


__all__ = [
    "request",
    "parse_body",
    "form",
    "endpoint",
    "html",
    "route",
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "bind_path",
    "bind_query",
    "bind_header",
]
