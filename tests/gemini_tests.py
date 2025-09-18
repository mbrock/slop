import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import rich

from slop import gemini
from slop.gemini import (
    FileState,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiError,
    GenerationConfig,
    ModelOverloadedError,
    ThinkingConfig,
    Tool,
    ToolConfig,
)
from slop.promptflow import ConversationBuilder, file as flow_file, line, text

from .testing import scope, spawn, test


@test
async def test_basic_text_generation() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("Write a haiku about Python programming.")

    response = await convo.complete(
        generation_config=GenerationConfig(temperature=0.7, maxOutputTokens=200)
    )
    assert response.candidates
    rich.print(response.candidates[0])  # DEBUG
    assert response.candidates[0].content.parts

    text_part = response.candidates[0].content.parts[0].text
    assert text_part is not None


@test
async def test_streaming_response() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("Count from 1 to 5.")

    request = await convo.to_request()

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

    with ConversationBuilder() as convo:
        with convo.user():
            line("What's the weather like in Tokyo?")

    response = await convo.complete(tools=[tool], tool_config=tool_config)
    assert response.candidates

    part = response.candidates[0].content.parts[0]
    assert part.functionCall or part.text


@test
async def test_multi_turn_conversation() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("My name is Alice. Remember it.")
        with convo.model():
            line("I'll remember that your name is Alice.")
        with convo.user():
            line("What's my name?")

    response = await convo.complete()
    assert response.candidates

    answer = response.candidates[0].content.parts[0].text
    assert answer is not None and "Alice" in answer


@test
async def test_file_upload_and_processing() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
        temp.write("Line 1\nLine 2\nLine 3")
        temp_path = Path(temp.name)

    uploaded = await gemini.upload_file(temp_path, display_name="Test File")
    assert uploaded.name
    assert uploaded.uri
    assert uploaded.state == FileState.ACTIVE

    try:
        with ConversationBuilder() as convo:
            with convo.user():
                line("Count the lines in this file.")
                flow_file(uploaded.uri)

        response = await convo.complete()
        assert response.candidates
    finally:
        await gemini.delete_file(uploaded.name)
        temp_path.unlink(missing_ok=True)


@test
async def test_invalid_model() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("Hello")

    with gemini.model.using("invalid-model-name"):
        try:
            await convo.complete()
        except GeminiError as exc:
            message = str(exc).lower()
            assert "not found" in message or "invalid" in message
        else:
            raise AssertionError("Expected GeminiError for invalid model name")


@test
async def test_long_prompt() -> None:
    long_text = "Tell me about " + " ".join(["artificial intelligence"] * 10000)

    with ConversationBuilder() as convo:
        with convo.user():
            line(long_text)

    try:
        response = await convo.complete(
            generation_config=GenerationConfig(maxOutputTokens=10)
        )
        assert response.candidates
    except GeminiError:
        pass


@test
async def test_model_overload_recovery() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("Quick test")

    try:
        response = await convo.complete()
        assert response.candidates
    except ModelOverloadedError as exc:
        assert exc.alternative_model and "flash" in exc.alternative_model
        with gemini.model.using(exc.alternative_model):
            response = await convo.complete()
        assert response.candidates


@test
async def test_thinking_basic() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            text(
                """Solve this step-by-step problem:

            A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
            Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
            The stations are 280 miles apart.

            At what time will the trains meet? Show your reasoning step by step."""
            )

    response = await convo.complete(
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        )
    )
    assert response.candidates

    parts = response.candidates[0].content.parts
    thought_parts = [p for p in parts if p.thought]
    regular_parts = [p for p in parts if not p.thought]
    assert thought_parts
    assert regular_parts
    assert all(part.text for part in thought_parts)


@test
async def test_thinking_disabled() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("What is 2 + 2?")

    response = await convo.complete(
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=False, thinkingBudget=0)
        )
    )
    assert response.candidates

    for part in response.candidates[0].content.parts:
        assert not part.thought


@test
async def test_thinking_dynamic() -> None:
    with ConversationBuilder() as convo:
        with convo.user():
            line("Come up with a mildly interesting fact. Think, but not too hard.")

    response = await convo.complete(
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=-1)
        )
    )
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

    with ConversationBuilder() as convo:
        with convo.user():
            text(
                """Calculate the compound interest on $10,000 at 5% annual rate for 3 years,
                           compounded monthly. Use the calculate function and explain your approach."""
            )

    response = await convo.complete(
        tools=[tool],
        tool_config=tool_config,
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )
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
    with ConversationBuilder() as first_turn:
        with first_turn.user():
            line("What factors should we consider when designing a sustainable city?")

    response1 = await first_turn.complete(
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        )
    )
    assert response1.candidates

    model_content = response1.candidates[0].content

    with ConversationBuilder() as follow_up:
        with follow_up.user():
            line("What factors should we consider when designing a sustainable city?")
        follow_up.append(model_content)
        with follow_up.user():
            line("Based on those factors, what would be the top 3 priorities?")

    response2 = await follow_up.complete(
        generation_config=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        )
    )
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
