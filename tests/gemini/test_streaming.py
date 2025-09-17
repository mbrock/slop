"""Test streaming responses from Gemini API."""

from src.slop import gemini
from src.slop.gemini import Content, GenerateRequest, Part


async def test_streaming_response():
    """Test streaming text generation."""
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="Count from 1 to 5.")])
    )

    full_text = ""
    chunk_count = 0

    async for response in gemini.generate_content(request):
        if response.candidates and response.candidates[0].content.parts:
            chunk = response.candidates[0].content.parts[0].text
            if chunk:
                full_text += chunk
                chunk_count += 1

    assert len(full_text) > 0
    assert chunk_count > 0
