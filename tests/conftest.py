"""Pytest configuration and fixtures."""

import os

import httpx
import pytest
import pytest_asyncio

from src.slop import gemini


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


@pytest_asyncio.fixture(scope="session", autouse=True)
async def configure_gemini_context():
    """Bind shared Gemini context for the entire test session."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.exit("GOOGLE_API_KEY environment variable not set", 1)

    default_model = "gemini-2.5-flash-lite"

    async with httpx.AsyncClient() as client:
        with gemini.http_client.using(client):
            with gemini.api_key.using(api_key):
                with gemini.model.using(default_model):
                    yield
