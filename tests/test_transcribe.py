import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from slop import store
from slop.app import create_app
from slop.models import list_interviews
from slop.parameter import Parameter

from .testing import test

current_client = Parameter[httpx.AsyncClient]("client")
database_paths = Parameter[tuple[str, str]]("db_paths")


@asynccontextmanager
async def appclient():
    """Provide an ASGI test client wired to temporary databases."""

    api_key = os.environ.get("GOOGLE_API_KEY")

    with tempfile.TemporaryDirectory() as data_dir:
        db1 = os.path.join(data_dir, "interviews.db")
        db2 = os.path.join(data_dir, "blobs.db")

        with store.with_databases(db1, db2):
            store.init_databases()

        async with httpx.AsyncClient() as their_client:
            app = create_app(
                {
                    "client": their_client,
                    "google_api_key": api_key,
                    "gemini_model": "gemini-2.5-flash-lite",
                    "interviews_db_path": db1,
                    "blobs_db_path": db2,
                }
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test.example",
            ) as client:
                with (
                    current_client.using(client),
                    database_paths.using((db1, db2)),
                    store.with_databases(db1, db2),
                ):
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
@appclient()
async def test_upload_audio_creates_interview():
    audio_path = Path("tests/data/kingcharles.mp3")

    client = current_client.get()
    audio_bytes = audio_path.read_bytes()
    response = await client.post(
        "/upload",
        files={"audio": ("kingcharles.mp3", audio_bytes, "audio/mpeg")},
    )

    assert response.status_code == 200

    interviews = list_interviews()
    assert len(interviews) == 1
    interview = interviews[0]

    assert interview.filename == "kingcharles.mp3"
    assert interview.audio_hash
    assert interview.duration == "00:05:09"

    blob, mime = store.get_blob(interview.audio_hash)
    assert mime == "audio/ogg"
    assert len(blob) > 0


@test
async def main():
    await test_home_endpoint()
    await test_interview_list_endpoint()
    await test_nonexistent_interview()
    await test_nonexistent_audio()
    await test_upload_audio_creates_interview()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
