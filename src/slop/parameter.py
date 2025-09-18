"""Shared Parameter class for context-based configuration."""

import os
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Self


class Parameter[T]:
    """Wrapper around :class:`contextvars.ContextVar` with convenience helpers."""

    _var: ContextVar[T]

    def __init__(
        self: Self,
        name: str,
    ):
        self._var: ContextVar[T] = ContextVar(name)

    def get(self: Self) -> T:
        return self._var.get()

    def peek(self: Self) -> T | None:
        return self._var.get(None)

    @contextmanager
    def using(self: Self, value: T):
        """Context manager to temporarily set the parameter value."""
        print("Setting parameter", self._var, value)
        token = self._var.set(value)
        try:
            yield
        finally:
            print("Resetting parameter", self._var, token)
            self._var.reset(token)

    @asynccontextmanager
    async def ausing(self: Self, value: T):
        """Async context manager to temporarily set the parameter value."""
        print("Setting parameter", self._var, value)
        token = self._var.set(value)
        try:
            yield
        finally:
            print("Resetting parameter", self._var, token)
            self._var.reset(token)

    @contextmanager
    def using_env(self: Self, env_var: str):
        """Context manager to set the parameter from an environment variable."""
        value = os.getenv(env_var)
        if value is None:
            raise LookupError(f"missing ${env_var} in environment")

        with self.using(value):
            yield
