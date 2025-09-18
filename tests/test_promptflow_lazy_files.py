import tempfile
from contextlib import asynccontextmanager

from slop import gemini
from slop.gemini import File, FileState
from slop.promptflow import current_contents, file, from_user, new_chat
from slop.store import init_databases, save_blob, with_databases

from .testing import test


@asynccontextmanager
async def temp_databases():
    """Provide isolated interview/blob databases for tests that touch storage."""

    with (
        tempfile.NamedTemporaryFile(suffix=".db") as interviews_tmp,
        tempfile.NamedTemporaryFile(suffix=".db") as blobs_tmp,
    ):
        with with_databases(interviews_tmp.name, blobs_tmp.name):
            init_databases()
            yield


@test
@temp_databases()
async def test_build_contents_uploads_blob_once() -> None:
    data = b"audio-bytes"
    blob_hash = save_blob(data, "audio/ogg")
    uploads: list[tuple[bytes, str]] = []

    async def fake_upload(payload: bytes, mime_type: str) -> File:
        uploads.append((payload, mime_type))
        return File(
            name=f"files/{blob_hash}",
            uri=f"files/{blob_hash}",
            mimeType=mime_type,
            state=FileState.ACTIVE,
        )

    with gemini.uploader.using(fake_upload):
        with new_chat():
            with from_user():
                file(f"blob:{blob_hash}", mime_type="audio/ogg")
                file(f"blob:{blob_hash}", mime_type="audio/ogg")

            contents = current_contents.get()
            # Resolve blob URIs
            await gemini.resolve_blob_uris(contents)

    assert uploads == [(data, "audio/ogg")]

    file_parts = [
        part for content in contents for part in content.parts if part.fileData
    ]
    assert len(file_parts) == 2
    for part in file_parts:
        file_data = part.fileData
        assert file_data is not None
        assert file_data.fileUri == f"files/{blob_hash}"
        assert file_data.mimeType == "audio/ogg"


async def main() -> None:
    await test_build_contents_uploads_blob_once()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
