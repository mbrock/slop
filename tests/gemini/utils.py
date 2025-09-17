"""Test utilities for Gemini tests."""

import pytest
from src.slop.gemini import ModelOverloadedError, GeminiError


def skip_if_overloaded(func):
    """Decorator to skip test if model is overloaded."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ModelOverloadedError:
            pytest.skip("Model is currently overloaded")
        except GeminiError as e:
            if "overloaded" in str(e).lower():
                pytest.skip("Model is currently overloaded")
            raise
    return wrapper