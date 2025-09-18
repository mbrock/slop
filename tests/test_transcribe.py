"""Basic integration tests for the transcribe.py web app."""

import os
import tempfile
from contextlib import asynccontextmanager

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
        api_key = os.getenv("GOOGLE_API_KEY", "test-api-key")

        # Use temporary files for test databases
        with tempfile.NamedTemporaryFile(suffix=".db") as interviews_tmp:
            with tempfile.NamedTemporaryFile(suffix=".db") as blobs_tmp:
                interviews_db_path = interviews_tmp.name
                blobs_db_path = blobs_tmp.name
                
                # Initialize databases (creates tables if needed)
                with models.sqlite_connection(interviews_db_path) as conn:
                    with models.interviews_db.using(conn):
                        with models.sqlite_connection(blobs_db_path) as blobs_conn:
                            with models.blobs_db.using(blobs_conn):
                                models.init_databases()
                
                async with httpx.AsyncClient() as client:
                    this.state.state = AppState(
                        client=client,
                        google_api_key=api_key,
                        gemini_model="gemini-2.5-flash-lite",
                        interviews_db_path=interviews_db_path,
                        blobs_db_path=blobs_db_path,
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
