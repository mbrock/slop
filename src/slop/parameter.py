"""Shared Parameter class for context-based configuration."""

import logging
from contextvars import ContextVar
from typing import Callable, TypeVar

logger = logging.getLogger("parameter")

T = TypeVar("T")


class Parameter[T]:
    """Wrapper around :class:`contextvars.ContextVar` with convenience helpers."""

    def __init__(
        self,
        name: str,
        *,
        default_factory: Callable[[], T] | None = None,
    ):
        self._var: ContextVar[T] = ContextVar(name)
        self._default_factory = default_factory

    def get(self) -> T:
        try:
            return self._var.get()
        except LookupError as exc:
            if self._default_factory is None:
                raise RuntimeError(f"Parameter {self._var.name} not set") from exc
            return self._default_factory()

    def using(self, value: T):
        """Context manager to temporarily set the parameter value."""
        print(f"Setting parameter {self._var.name} to {value}")
        token = self._var.set(value)
        var = self._var

        class ParameterContext:
            def __enter__(self):
                return value

            def __exit__(self, *args):
                print(f"Resetting parameter {var.name} to previous value")
                var.reset(token)

        return ParameterContext()
