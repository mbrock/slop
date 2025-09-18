import tempfile
from contextlib import asynccontextmanager

from slop.gemini import File, FileState
from slop.models import init_databases, save_blob, with_databases
from slop.promptflow import ConversationBuilder, file

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

    conversation = ConversationBuilder(upload=fake_upload)
    with conversation.user_turn():
        file(f"blob:{blob_hash}", mime_type="audio/ogg")
        file(f"blob:{blob_hash}", mime_type="audio/ogg")

    contents = await conversation.build_contents()

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


@test
@temp_databases()
async def test_build_contents_conflicting_mime_types() -> None:
    blob_hash = save_blob(b"audio-bytes", "audio/ogg")

    async def fail_upload(
        *_: object, **__: object
    ) -> File:  # pragma: no cover - defensive
        raise AssertionError("upload should not be invoked")

    conversation = ConversationBuilder(upload=fail_upload)
    with conversation.user_turn():
        file(f"blob:{blob_hash}", mime_type="audio/ogg")
        file(f"blob:{blob_hash}", mime_type="audio/wav")

    try:
        await conversation.build_contents()
    except ValueError as e:
        assert "MIME" in str(e)
    else:
        assert False, "Expected ValueError for conflicting MIME types"


@test
async def main() -> None:
    await test_build_contents_uploads_blob_once()
    await test_build_contents_conflicting_mime_types()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
