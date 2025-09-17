"""Test function calling with Gemini API."""

import pytest

from src.slop.gemini import (
    Content,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiClient,
    GenerateRequest,
    Part,
    Tool,
    ToolConfig,
)


@pytest.mark.asyncio
async def test_function_calling():
    """Test function calling feature."""
    client = GeminiClient()

    # Define a function for getting weather
    weather_function = FunctionDeclaration(
        name="get_weather",
        description="Get the weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )

    tool = Tool(functionDeclarations=[weather_function])
    tool_config = ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))

    request = GenerateRequest(
        contents=Content(
            role="user", parts=[Part(text="What's the weather like in Tokyo?")]
        ),
        tools=[tool],
        toolConfig=tool_config,
    )

    response = await client.generate_content_sync(
        request, model="gemini-2.5-flash-lite"
    )
    assert response.candidates

    # Model can either call the function or respond with text
    if response.candidates[0].content.parts:
        part = response.candidates[0].content.parts[0]
        # Either function call or text is valid
        assert part.functionCall or part.text
