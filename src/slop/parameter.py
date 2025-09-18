"""Shared Parameter class for context-based configuration."""

import os
from contextlib import contextmanager
from contextvars import ContextVar


class Parameter[T]:
    """Wrapper around :class:`contextvars.ContextVar` with convenience helpers."""

    def __init__(
        self,
        name: str,
    ):
        self._var: ContextVar[T] = ContextVar(name)

    def get(self) -> T:
        return self._var.get()

    @contextmanager
    def using(self, value: T):
        """Context manager to temporarily set the parameter value."""
        token = self._var.set(value)
        try:
            yield
        finally:
            self._var.reset(token)

    @contextmanager
    def using_env(self, env_var: str):
        """Context manager to set the parameter from an environment variable."""
        value = os.getenv(env_var)
        if value is None:
            raise LookupError(f"missing ${env_var} in environment")

        with self.using(value):
            yield
