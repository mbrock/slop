"""Test streaming responses from Gemini API."""

import pytest

from src.slop.gemini import Content, GeminiClient, GenerateRequest, Part


@pytest.mark.asyncio
async def test_streaming_response():
    """Test streaming text generation."""
    client = GeminiClient()

    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="Count from 1 to 5.")])
    )

    full_text = ""
    chunk_count = 0

    async for response in client.generate_content(
        request, model="gemini-2.5-flash-lite"
    ):
        if response.candidates and response.candidates[0].content.parts:
            chunk = response.candidates[0].content.parts[0].text
            if chunk:
                full_text += chunk
                chunk_count += 1

    assert len(full_text) > 0
    assert chunk_count > 0
