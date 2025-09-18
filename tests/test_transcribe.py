import os
import tempfile
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

import slop.models as models
from slop.app import AppState, configure_app
from slop.parameter import Parameter

from .testing import test

current_client = Parameter[httpx.AsyncClient]("client")


def build_app() -> FastAPI:
    """Create a fresh FastAPI application for testing."""

    return configure_app(FastAPI(title="Ieva's Interviews Test"))


@asynccontextmanager
async def appclient():
    """Provide an ASGI test client wired to temporary databases."""

    api_key = os.getenv("GOOGLE_API_KEY", "test-api-key")
    app = build_app()

    with (
        tempfile.NamedTemporaryFile(suffix=".db") as interviews_tmp,
        tempfile.NamedTemporaryFile(suffix=".db") as blobs_tmp,
    ):
        interviews_db_path = interviews_tmp.name
        blobs_db_path = blobs_tmp.name

        with models.with_databases(interviews_db_path, blobs_db_path):
            models.init_databases()

            async with httpx.AsyncClient() as upstream_client:
                app.state.state = AppState(
                    client=upstream_client,
                    google_api_key=api_key,
                    gemini_model="gemini-2.5-flash-lite",
                    interviews_db_path=interviews_db_path,
                    blobs_db_path=blobs_db_path,
                )

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test.example",
                ) as client:
                    with current_client.using(client):
                        yield client


async def get(url: str):
    client = current_client.get()
    return await client.get(url)


@test
@appclient()
async def test_home_endpoint():
    response = await get("/")
    assert response.status_code == 200
    assert "Ieva's Interviews" in response.text


@test
@appclient()
async def test_interview_list_endpoint():
    response = await get("/interview-list")
    assert response.status_code == 200
    assert "interview-list" in response.text


@test
@appclient()
async def test_nonexistent_interview():
    response = await get("/interview/nonexistent")
    assert response.status_code == 404


@test
@appclient()
async def test_nonexistent_audio():
    response = await get("/audio/nonexistent")
    assert response.status_code == 404


@test
async def main():
    await test_home_endpoint()
    await test_interview_list_endpoint()
    await test_nonexistent_interview()
    await test_nonexistent_audio()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
