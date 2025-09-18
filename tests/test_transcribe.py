"""Basic integration tests for the transcribe.py web app."""

import os
import tempfile

import httpx
import pytest
from fastapi import FastAPI

import src.slop.models as models
from src.slop.app import AppState, configure_app


@pytest.fixture
def app():
    return configure_app(FastAPI(title="Ieva's Interviews Test"))


@pytest.fixture
async def client(app: FastAPI):
    api_key = os.getenv("GOOGLE_API_KEY", "test-api-key")

    with (
        tempfile.NamedTemporaryFile(suffix=".db") as interviews_tmp,
        tempfile.NamedTemporaryFile(suffix=".db") as blobs_tmp,
    ):
        interviews_db_path = interviews_tmp.name
        blobs_db_path = blobs_tmp.name

        with models.with_databases(interviews_db_path, blobs_db_path):
            models.init_databases()

        async with httpx.AsyncClient() as their_client:
            app.state.state = AppState(
                client=their_client,
                google_api_key=api_key,
                gemini_model="gemini-2.5-flash-lite",
                interviews_db_path=interviews_db_path,
                blobs_db_path=blobs_db_path,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test.example",
            ) as client:
                print("Starting test client")
                yield client
                print("Stopping test client")


async def test_home_endpoint(client):
    print("Testing home endpoint")
    response = await client.get("/")
    assert response.status_code == 200
    assert "Ieva's Interviews" in response.text


async def test_interview_list_endpoint(client):
    response = await client.get("/interview-list")
    assert response.status_code == 200
    assert "interview-list" in response.text


async def test_nonexistent_interview(client):
    response = await client.get("/interview/nonexistent")
    assert response.status_code == 404


async def test_nonexistent_audio(client):
    response = await client.get("/audio/nonexistent")
    assert response.status_code == 404
