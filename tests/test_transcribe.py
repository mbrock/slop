"""Basic integration tests for the transcribe.py web app."""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.slop.models as models
from src.slop.app import AppState, configure_app


@pytest.fixture
def app():
    @asynccontextmanager
    async def lifespan(this):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable must be set")

        with tempfile.TemporaryDirectory() as tmpdir:
            with models.sqlite(Path(tmpdir), "interviews.db") as interviews_conn:
                with models.interviews_db.using(interviews_conn):
                    with models.sqlite(Path(tmpdir), "blobs.db") as blobs_conn:
                        with models.blobs_db.using(blobs_conn):
                            models.init_databases()
                            async with httpx.AsyncClient() as client:
                                this.state.state = AppState(
                                    client=client,
                                    google_api_key=api_key,
                                    gemini_model="gemini-2.5-flash-lite",
                                    interview_db=interviews_conn,
                                    blobs_db=blobs_conn,
                                )
                                yield

    return configure_app(
        FastAPI(
            title="Ieva's Interviews Test",
            lifespan=lifespan,
        )
    )


@pytest.fixture
def client(app: FastAPI):
    """Async HTTP client for testing the FastAPI app."""

    with TestClient(app) as client:
        print("Starting test client")
        yield client
        print("Stopping test client")


def test_home_endpoint(client):
    """Test that the home endpoint returns successfully."""
    print("Testing home endpoint")
    response = client.get("/")
    assert response.status_code == 200
    assert "Ieva's Interviews" in response.text


def test_interview_list_endpoint(client):
    """Test that the interview list endpoint returns successfully."""
    response = client.get("/interview-list")
    assert response.status_code == 200
    assert "interview-list" in response.text


def test_nonexistent_interview(client):
    """Test that accessing a non-existent interview returns 404."""
    response = client.get("/interview/nonexistent")
    assert response.status_code == 404


def test_nonexistent_audio(client):
    """Test that accessing non-existent audio returns 404."""
    response = client.get("/audio/nonexistent")
    assert response.status_code == 404
