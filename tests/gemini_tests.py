import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.abc
import rich

from slop import gemini
from slop.gemini import (
    Content,
    FileData,
    FileState,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiError,
    GenerateRequest,
    GenerationConfig,
    ModelOverloadedError,
    Part,
    ThinkingConfig,
    Tool,
    ToolConfig,
)

from .testing import scope, spawn, test


@test
async def test_basic_text_generation() -> None:
    request = GenerateRequest(
        contents=Content(
            role="user", parts=[Part(text="Write a haiku about Python programming.")]
        ),
        generationConfig=GenerationConfig(temperature=0.7, maxOutputTokens=200),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates
    rich.print(response.candidates[0])  # DEBUG
    assert response.candidates[0].content.parts

    text = response.candidates[0].content.parts[0].text
    assert text is not None


@test
async def test_streaming_response() -> None:
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

    assert chunk_count > 0
    assert len(full_text) > 0


@test
async def test_list_files() -> None:
    file_list = await gemini.list_files(page_size=5)
    assert isinstance(file_list.files, list)

    if file_list.files:
        file = file_list.files[0]
        assert hasattr(file, "name")
        assert hasattr(file, "uri")


@test
async def test_function_calling() -> None:
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

    response = await gemini.generate_content_sync(request)
    assert response.candidates

    part = response.candidates[0].content.parts[0]
    assert part.functionCall or part.text


@test
async def test_multi_turn_conversation() -> None:
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

    text = response.candidates[0].content.parts[0].text
    assert text is not None and "Alice" in text


@test
async def test_file_upload_and_processing() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Line 1\nLine 2\nLine 3")
        temp_path = Path(f.name)

    uploaded = await gemini.upload_file(temp_path, display_name="Test File")
    assert uploaded.name
    assert uploaded.uri
    assert uploaded.state == FileState.ACTIVE

    try:
        request = GenerateRequest(
            contents=Content(
                role="user",
                parts=[
                    Part(text="Count the lines in this file."),
                    Part(fileData=FileData(fileUri=uploaded.uri)),
                ],
            )
        )

        response = await gemini.generate_content_sync(request)
        assert response.candidates
    finally:
        await gemini.delete_file(uploaded.name)
        temp_path.unlink(missing_ok=True)


@test
async def test_invalid_model() -> None:
    request = GenerateRequest(contents=Content(role="user", parts=[Part(text="Hello")]))

    with gemini.model.using("invalid-model-name"):
        try:
            await gemini.generate_content_sync(request)
        except GeminiError as exc:
            message = str(exc).lower()
            assert "not found" in message or "invalid" in message
        else:
            raise AssertionError("Expected GeminiError for invalid model name")


@test
async def test_long_prompt() -> None:
    long_text = "Tell me about " + " ".join(["artificial intelligence"] * 10000)
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text=long_text)]),
        generationConfig=GenerationConfig(maxOutputTokens=10),
    )

    try:
        response = await gemini.generate_content_sync(request)
        assert response.candidates
    except GeminiError:
        pass


@test
async def test_model_overload_recovery() -> None:
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="Quick test")])
    )

    try:
        response = await gemini.generate_content_sync(request)
        assert response.candidates
    except ModelOverloadedError as exc:
        assert exc.alternative_model and "flash" in exc.alternative_model
        with gemini.model.using(exc.alternative_model):
            response = await gemini.generate_content_sync(request)
        assert response.candidates


@test
async def test_thinking_basic() -> None:
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[
                Part(
                    text="""Solve this step-by-step problem:

            A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
            Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
            The stations are 280 miles apart.

            At what time will the trains meet? Show your reasoning step by step."""
                )
            ],
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates

    parts = response.candidates[0].content.parts
    thought_parts = [p for p in parts if p.thought]
    regular_parts = [p for p in parts if not p.thought]
    assert thought_parts
    assert regular_parts
    assert all(part.text for part in thought_parts)


@test
async def test_thinking_disabled() -> None:
    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="What is 2 + 2?")]),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=False, thinkingBudget=0)
        ),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates

    for part in response.candidates[0].content.parts:
        assert not part.thought


@test
async def test_thinking_dynamic() -> None:
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[
                Part(
                    text="Come up with a mildly interesting fact. Think, but not too hard."
                ),
            ],
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=-1)
        ),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates

    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert thought_parts


@test
async def test_thinking_with_function_calling() -> None:
    math_function = FunctionDeclaration(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    )

    tool = Tool(functionDeclarations=[math_function])
    tool_config = ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))

    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[
                Part(
                    text="""Calculate the compound interest on $10,000 at 5% annual rate for 3 years,
                           compounded monthly. Use the calculate function and explain your approach."""
                )
            ],
        ),
        tools=[tool],
        toolConfig=tool_config,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )

    response = await gemini.generate_content_sync(request)
    assert response.candidates

    parts = response.candidates[0].content.parts
    thought_parts = [p for p in parts if p.thought]
    function_parts = [p for p in parts if p.functionCall]
    signature_parts = [p for p in parts if p.thoughtSignature]
    assert thought_parts
    assert function_parts
    assert signature_parts


@test
async def test_thinking_multi_turn_with_signatures() -> None:
    request1 = GenerateRequest(
        contents=Content(
            role="user",
            parts=[
                Part(
                    text="What factors should we consider when designing a sustainable city?"
                )
            ],
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )

    response1 = await gemini.generate_content_sync(request1)
    assert response1.candidates

    conversation = [
        Content(
            role="user",
            parts=[
                Part(
                    text="What factors should we consider when designing a sustainable city?"
                )
            ],
        ),
        response1.candidates[0].content,
        Content(
            role="user",
            parts=[
                Part(text="Based on those factors, what would be the top 3 priorities?")
            ],
        ),
    ]

    request2 = GenerateRequest(
        contents=conversation,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )

    response2 = await gemini.generate_content_sync(request2)
    assert response2.candidates
    assert response2.candidates[0].content.parts


@asynccontextmanager
async def use_gemini():
    import httpx

    async with httpx.AsyncClient() as client:
        with (
            gemini.http_client.using(client),
            gemini.api_key.using_env("GOOGLE_API_KEY"),
            gemini.model.using("gemini-2.5-flash-lite"),
        ):
            yield


@test
@use_gemini()
@scope()
async def main():
    spawn(test_thinking_multi_turn_with_signatures)
    spawn(test_thinking_basic)
    spawn(test_thinking_dynamic)
    spawn(test_thinking_with_function_calling)

    spawn(test_file_upload_and_processing)
    spawn(test_basic_text_generation)
    spawn(test_streaming_response)
    spawn(test_list_files)
    spawn(test_function_calling)
    spawn(test_multi_turn_conversation)
    spawn(test_invalid_model)
    spawn(test_long_prompt)
    spawn(test_model_overload_recovery)
    spawn(test_thinking_disabled)


if __name__ == "__main__":
    import anyio

    anyio.run(main)
