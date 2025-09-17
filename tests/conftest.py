"""Pytest configuration and fixtures."""

import os

import pytest

from src.slop.gemini import GeminiClient


def pytest_configure(config: pytest.Config):
    """Configure pytest."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.exit("GOOGLE_API_KEY environment variable not set", 1)


def pytest_addoption(parser: pytest.Parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (marked with @pytest.mark.slow)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Modify collected test items to skip slow tests unless --runslow is given."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
async def gemini_client():
    """Fixture that provides a GeminiClient with gemini-2.5-flash-lite model."""
    client = GeminiClient(model="gemini-2.5-flash-lite")
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def gemini_client_pro():
    """Fixture that provides a GeminiClient with gemini-2.5-flash model."""
    client = GeminiClient(model="gemini-2.5-flash")
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def gemini_client_exp():
    """Fixture that provides a GeminiClient with gemini-2.0-flash-exp model."""
    client = GeminiClient(model="gemini-2.0-flash-exp")
    try:
        yield client
    finally:
        await client.close()
