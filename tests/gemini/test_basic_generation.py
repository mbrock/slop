"""Test basic text generation with Gemini API."""

from src.slop import gemini
from src.slop.gemini import Content, GenerateRequest, GenerationConfig, Part


async def test_basic_text_generation():
    """Test basic text generation with a simple prompt."""
    request = GenerateRequest(
        contents=Content(
            role="user", parts=[Part(text="Write a haiku about Python programming.")]
        ),
        generationConfig=GenerationConfig(temperature=0.7, maxOutputTokens=100),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates
    assert len(response.candidates) > 0

    # Text might be empty if model hits token limit
    if response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text
        assert text is not None
