"""Pytest configuration and fixtures."""

import os
import pytest


def pytest_configure(config):
    """Configure pytest."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.exit("GOOGLE_API_KEY environment variable not set", 1)