"""Test error handling for Gemini API."""

import pytest

from src.slop.gemini import (
    Content,
    GeminiClient,
    GeminiError,
    GenerateRequest,
    GenerationConfig,
    ModelOverloadedError,
    Part,
)


@pytest.mark.asyncio
async def test_invalid_model():
    """Test invalid model name error handling."""
    client = GeminiClient()

    request = GenerateRequest(contents=Content(role="user", parts=[Part(text="Hello")]))

    with pytest.raises(GeminiError) as exc_info:
        await client.generate_content_sync(request, model="invalid-model-name")

    assert (
        "not found" in str(exc_info.value).lower()
        or "invalid" in str(exc_info.value).lower()
    )


@pytest.mark.asyncio
async def test_long_prompt():
    """Test handling of very long prompts."""
    client = GeminiClient()

    # Create a prompt that exceeds typical token limits
    long_text = "Tell me about " + " ".join(["artificial intelligence"] * 10000)
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text=long_text)]),
        generationConfig=GenerationConfig(maxOutputTokens=10),
    )

    # Should either succeed or raise an error - both are acceptable
    try:
        response = await client.generate_content_sync(
            request, model="gemini-2.5-flash-lite"
        )
        assert response.candidates  # If it succeeds, should have candidates
    except GeminiError:
        pass  # Expected for very long prompts


@pytest.mark.asyncio
async def test_model_overload_recovery():
    """Test model overload error handling and recovery."""
    client = GeminiClient()

    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="Quick test")])
    )

    # This test documents the overload recovery pattern
    try:
        response = await client.generate_content_sync(
            request, model="gemini-2.5-flash-lite"
        )
        assert response.candidates  # Normal success
    except ModelOverloadedError as e:
        # If overloaded, should suggest alternative
        assert e.alternative_model
        assert "flash" in e.alternative_model
        # Try alternative
        response = await client.generate_content_sync(
            request, model=e.alternative_model
        )
        assert response.candidates
