"""Test multi-turn conversation with Gemini API."""

from src.slop import gemini
from src.slop.gemini import Content, GenerateRequest, Part


async def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    # Build conversation history
    contents = [
        Content(role="user", parts=[Part(text="My name is Alice. Remember it.")]),
        Content(
            role="model", parts=[Part(text="I'll remember that your name is Alice.")]
        ),
        Content(role="user", parts=[Part(text="What's my name?")]),
    ]

    request = GenerateRequest(contents=contents)
    response = await gemini.generate_content_sync(request)

    assert response.candidates
    assert len(response.candidates) > 0

    if response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text
        assert text is not None
        # Check if the model remembers the name
        assert "Alice" in text
