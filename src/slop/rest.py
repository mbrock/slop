"""
Generic REST framework utilities for Starlette applications.

Provides decorators, helpers, and middleware for building REST APIs with:
- Automatic form validation using TypedDict
- Context parameter management
- Response wrapping for HTML/HTMX endpoints
"""

import functools
import inspect
from typing import Type

from pydantic import ConfigDict, TypeAdapter, ValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Route
from tagflow import TagResponse

from slop.parameter import Parameter

# ============================================================================
# Context Parameters
# ============================================================================

request = Parameter[Request]("rest_request")


# ============================================================================
# Form Validation
# ============================================================================


async def form[T](model: Type[T]) -> T:
    """Get and validate form data against a TypedDict.

    Automatically converts string integers to int type where needed.

    Args:
        model: TypedDict class to validate against

    Returns:
        Validated form data as TypedDict instance

    Raises:
        HTTPException: If validation fails with details about errors
    """
    req = request.get()
    form_data = await req.form()

    config = getattr(model, "__pydantic_config__", None)
    if not getattr(config, "arbitrary_types_allowed", False):
        try:
            setattr(model, "__pydantic_config__", ConfigDict(arbitrary_types_allowed=True))
        except (AttributeError, TypeError):
            pass

    adapter = TypeAdapter(model)

    try:
        return adapter.validate_python(form_data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")
        raise HTTPException(
            status_code=400, detail="Validation error: " + "; ".join(errors)
        )


# ============================================================================
# Route Decorators
# ============================================================================


def html(func, model_type: Type | None = None):
    """Decorator that wraps handlers with common functionality.

    - Sets request context
    - Handles form validation if model_type is provided
    - Ensures TagResponse is returned for None results
    - Supports both async and sync functions
    """

    @functools.wraps(func)
    async def wrapper(req: Request, *args, **kwargs):
        with request.using(req):
            if model_type:
                form_data = await form(model_type)
                # Check if func is async
                if inspect.iscoroutinefunction(func):
                    response = await func(**form_data)
                else:
                    response = func(**form_data)
            else:
                # Check if func is async
                if inspect.iscoroutinefunction(func):
                    response = await func()
                else:
                    response = func()
            if not response:
                return TagResponse()
            return response

    return wrapper


def GET(path: str, endpoint):
    """Create a GET route with html decorator applied."""
    return Route(path, html(endpoint), methods=["GET"])


def POST(path: str, endpoint, form: Type | None = None):
    """Create a POST route with html decorator and optional form validation."""
    return Route(path, html(endpoint, model_type=form), methods=["POST"])


def PUT(path: str, endpoint, form: Type | None = None):
    """Create a PUT route with html decorator and optional form validation."""
    return Route(path, html(endpoint, model_type=form), methods=["PUT"])


# ============================================================================
# Middleware Utilities
# ============================================================================


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware that sets the request in context for all handlers."""

    async def dispatch(self, req, call_next):
        with request.using(req):
            return await call_next(req)


def load(param: Parameter, path_param: str, loader):
    """Create middleware that loads a value and sets it in a Parameter context.

    Args:
        param: The Parameter to set
        path_param: The path parameter name to extract
        loader: Function that takes the path param value and returns the context value
                Can be sync or async.

    Returns:
        A middleware class that can be used with Starlette
    """

    class ContextMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, req, call_next):
            value_id = req.path_params.get(path_param)

            # Load the value (handle both sync and async loaders)
            if inspect.iscoroutinefunction(loader):
                value = await loader(value_id)
            else:
                value = loader(value_id)

            # Set in context and continue
            with param.using(value):
                return await call_next(req)

    return ContextMiddleware
