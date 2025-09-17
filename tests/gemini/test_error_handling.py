"""Test error handling for Gemini API."""

import pytest

from src.slop import gemini
from src.slop.gemini import (
    Content,
    GeminiError,
    GenerateRequest,
    GenerationConfig,
    ModelOverloadedError,
    Part,
)


async def test_invalid_model():
    """Test invalid model name error handling."""

    request = GenerateRequest(contents=Content(role="user", parts=[Part(text="Hello")]))

    with gemini.model.using("invalid-model-name"):
        with pytest.raises(GeminiError) as exc_info:
            await gemini.generate_content_sync(request)

    assert (
        "not found" in str(exc_info.value).lower()
        or "invalid" in str(exc_info.value).lower()
    )


async def test_long_prompt():
    """Test handling of very long prompts."""
    # Create a prompt that exceeds typical token limits
    long_text = "Tell me about " + " ".join(["artificial intelligence"] * 10000)
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text=long_text)]),
        generationConfig=GenerationConfig(maxOutputTokens=10),
    )

    # Should either succeed or raise an error - both are acceptable
    try:
        response = await gemini.generate_content_sync(request)
        assert response.candidates  # If it succeeds, should have candidates
    except GeminiError:
        pass  # Expected for very long prompts


async def test_model_overload_recovery():
    """Test model overload error handling and recovery."""
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="Quick test")])
    )

    # This test documents the overload recovery pattern
    try:
        response = await gemini.generate_content_sync(request)
        assert response.candidates  # Normal success
    except ModelOverloadedError as e:
        # If overloaded, should suggest alternative
        assert e.alternative_model
        assert "flash" in e.alternative_model
        # Try alternative with a new client using the alternative model
        with gemini.model.using(e.alternative_model):
            response = await gemini.generate_content_sync(request)
        assert response.candidates
