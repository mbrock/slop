"""Test basic text generation with Gemini API."""

import pytest

from src.slop.gemini import (
    Content,
    GeminiClient,
    GenerateRequest,
    GenerationConfig,
    Part,
)


@pytest.mark.asyncio
async def test_basic_text_generation():
    """Test basic text generation with a simple prompt."""
    client = GeminiClient()

    request = GenerateRequest(
        contents=Content(
            role="user", parts=[Part(text="Write a haiku about Python programming.")]
        ),
        generationConfig=GenerationConfig(temperature=0.7, maxOutputTokens=100),
    )

    response = await client.generate_content_sync(
        request, model="gemini-2.5-flash-lite"
    )
    assert response.candidates
    assert len(response.candidates) > 0

    # Text might be empty if model hits token limit
    if response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text
        assert text is not None
