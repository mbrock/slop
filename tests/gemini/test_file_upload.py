"""Test file upload and processing with Gemini API."""

import tempfile
from pathlib import Path

import pytest

from src.slop.gemini import (
    Content,
    FileData,
    FileState,
    GeminiClient,
    GenerateRequest,
    Part,
)


@pytest.mark.asyncio
async def test_file_upload_and_processing():
    """Test file upload and processing."""
    client = GeminiClient()

    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Line 1\n")
        f.write("Line 2\n")
        f.write("Line 3")
        temp_path = Path(f.name)

    try:
        # Upload the file
        file = await client.upload_file(temp_path, display_name="Test File")
        assert file.name
        assert file.uri
        assert file.state == FileState.ACTIVE

        # Process the file
        request = GenerateRequest(
            contents=Content(
                role="user",
                parts=[
                    Part(text="Count the lines in this file."),
                    Part(fileData=FileData(fileUri=file.uri)),
                ],
            )
        )

        response = await client.generate_content_sync(
            request, model="gemini-2.5-flash-lite"
        )
        assert response.candidates

        # Clean up
        await client.delete_file(file.name)

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)
