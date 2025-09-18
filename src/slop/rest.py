"""Lightweight request helpers shared across the application."""

from typing import Type, TypeVar

from pydantic import ConfigDict, TypeAdapter, ValidationError
from starlette.exceptions import HTTPException
from starlette.requests import Request

from slop.parameter import Parameter

T = TypeVar("T")

# ============================================================================
# Context Parameters
# ============================================================================

request = Parameter[Request]("rest_request")


# ============================================================================
# Form Validation
# ============================================================================


async def form(model: Type[T]) -> T:
    """Get and validate form data against a TypedDict."""

    req = request.get()
    form_data = await req.form()

    config = getattr(model, "__pydantic_config__", None)
    if not getattr(config, "arbitrary_types_allowed", False):
        try:
            setattr(
                model, "__pydantic_config__", ConfigDict(arbitrary_types_allowed=True)
            )
        except (AttributeError, TypeError):
            pass

    adapter = TypeAdapter(model)

    try:
        return adapter.validate_python(form_data)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")
        raise HTTPException(
            status_code=400, detail="Validation error: " + "; ".join(errors)
        )
