import os
import sys
import tempfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import httpx

from slop import store
from slop.app import create_app
from slop.models import list_tapes
from slop.parameter import Parameter

from .testing import test, test_filter

current_client = Parameter[httpx.AsyncClient]("client")


@asynccontextmanager
async def appclient():
    """Provide an ASGI test client wired to temporary databases."""

    api_key = os.environ.get("GOOGLE_API_KEY")

    with tempfile.TemporaryDirectory() as data_dir:
        db1 = os.path.join(data_dir, "tapes.db")
        db2 = os.path.join(data_dir, "blobs.db")

        with store.with_databases(db1, db2):
            store.init_databases()

        async with httpx.AsyncClient() as their_client:
            app = create_app(
                {
                    "client": their_client,
                    "google_api_key": api_key,
                    "gemini_model": "gemini-2.5-flash-lite",
                    "tapes_db_path": db1,
                    "blobs_db_path": db2,
                }
            )

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test.example",
            ) as client:
                with (
                    current_client.using(client),
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
    assert "Ieva's Tapes" in response.text


@test
@appclient()
async def test_tape_list_endpoint():
    response = await get("/tapes?partial=list")
    assert response.status_code == 200
    assert "tape-list" in response.text


@test
@appclient()
async def test_nonexistent_tape():
    response = await get("/tapes/nonexistent")
    assert response.status_code == 404


@test
@appclient()
async def test_nonexistent_audio():
    response = await get("/audio/nonexistent")
    assert response.status_code == 404


@test
@appclient()
async def test_upload_audio_creates_tape():
    audio_path = Path("tests/data/kingcharles.mp3")

    client = current_client.get()
    audio_bytes = audio_path.read_bytes()
    response = await client.post(
        "/tapes",
        files={"audio": ("kingcharles.mp3", audio_bytes, "audio/mpeg")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "id" in data

    tapes = list_tapes()
    assert len(tapes) == 1
    tape = tapes[0]

    assert tape.filename == "kingcharles.mp3"
    assert tape.audio_hash
    assert tape.duration == "00:05:09"

    blob, mime = store.get_blob(tape.audio_hash)
    assert mime == "audio/ogg"
    assert len(blob) > 0


@test
@appclient()
async def test_transcribe_next_part_integration():
    client = current_client.get()
    audio_path = Path("tests/data/kingcharles.mp3")
    audio_bytes = audio_path.read_bytes()

    upload_response = await client.post(
        "/tapes",
        files={"audio": ("kingcharles.mp3", audio_bytes, "audio/mpeg")},
    )
    assert upload_response.status_code == 201
    data = upload_response.json()
    assert "id" in data

    tapes = list_tapes()
    assert tapes, "Upload did not create any tapes"
    tape_id = tapes[-1].id

    response = await client.post(
        f"/tapes/{tape_id}/jobs",
        data={
            "kind": "transcribe_next",
            "duration_seconds": "10",
            "model_name": "gemini-2.5-flash-lite",
        },
    )

    assert response.status_code == 200, response.text

    tapes = list_tapes()
    tape = next((i for i in tapes if i.id == tape_id), None)
    assert tape is not None
    assert tape.parts

    part = tape.parts[-1]
    assert part.start_time == "00:00:00"
    assert part.end_time == "00:00:10"
    assert part.utterances


async def main():
    filter_value = os.environ.get("TEST_FILTER")
    if len(sys.argv) > 1:
        filter_value = sys.argv[1]

    context = test_filter.using(filter_value) if filter_value else nullcontext()

    with context:
        await test_home_endpoint()
        await test_tape_list_endpoint()
        await test_nonexistent_tape()
        await test_nonexistent_audio()
        await test_upload_audio_creates_tape()
        await test_transcribe_next_part_integration()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
