import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from slop import gemini, conf
from slop.gemini import (
    FileState,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiError,
    Tool,
    ToolConfig,
)
from slop.promptflow import (
    file as flow_file,
)
from slop.promptflow import (
    from_model,
    from_user,
    generate,
    generate_streaming,
    line,
    new_chat,
    text,
)

from .testing import scope, spawn, test


@test
async def test_basic_text_generation() -> None:
    with (
        conf.temperature(0.7),
        conf.max_output(200),
        new_chat(),
    ):
        with from_user():
            line("Write a haiku about Python programming.")

        response = await generate()

        assert response.candidates
        assert response.candidates[0].content.parts

        text_part = response.candidates[0].content.parts[0].text
        assert text_part is not None


@test
async def test_streaming_response() -> None:
    with conf.thinking(include_thoughts=True, budget=512):
        with new_chat():
            with from_user():
                line("Come up with a limerick about an obscure topic.")

            full_text = ""
            chunk_count = 0

            async for response in generate_streaming():
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

    with new_chat():
        with from_user():
            line("What's the weather like in Tokyo?")

        with conf.tools(tool), conf.tool_config("ANY"):
            response = await generate()
        assert response.candidates

        part = response.candidates[0].content.parts[0]
        assert part.functionCall or part.text


@test
async def test_multi_turn_conversation() -> None:
    with new_chat():
        with from_user():
            line("My name is Alice. Remember it.")
        with from_model():
            line("I'll remember that your name is Alice.")
        with from_user():
            line("What's my name?")

        response = await generate()
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
        with new_chat():
            with from_user():
                line("Count the lines in this file.")
                flow_file(uploaded.uri)

            response = await generate()
            assert response.candidates
    finally:
        await gemini.delete_file(uploaded.name)
        temp_path.unlink(missing_ok=True)


@test
async def test_invalid_model() -> None:
    with new_chat():
        with from_user():
            line("Hello")

        with gemini.model.using("invalid-model-name"):
            try:
                await generate()
            except GeminiError as exc:
                message = str(exc).lower()
                assert "not found" in message or "invalid" in message
            else:
                raise AssertionError("Expected GeminiError for invalid model name")


@test
async def test_thinking_basic() -> None:
    with new_chat():
        with from_user():
            text(
                """Solve this step-by-step problem:

            A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
            Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
            The stations are 280 miles apart.

            At what time will the trains meet? Show your reasoning step by step."""
            )

        with conf.thinking(include_thoughts=True, budget=512):
            response = await generate()

    assert response.candidates

    parts = response.candidates[0].content.parts
    thought_parts = [p for p in parts if p.thought]
    regular_parts = [p for p in parts if not p.thought]
    assert thought_parts
    assert regular_parts
    assert all(part.text for part in thought_parts)


@test
async def test_thinking_disabled() -> None:
    with new_chat():
        with from_user():
            line("What is 2 + 2?")

        with conf.thinking(include_thoughts=False, budget=0):
            response = await generate()

            assert response.candidates

            for part in response.candidates[0].content.parts:
                assert not part.thought


@test
async def test_thinking_dynamic() -> None:
    with new_chat():
        with from_user():
            line("Come up with a mildly interesting fact. Think, but not too hard.")

        with conf.thinking(include_thoughts=True, budget=-1):
            response = await generate()
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

    with new_chat():
        with from_user():
            text(
                """Calculate the compound interest on $10,000 at 5% annual rate for 3 years,
                           compounded monthly. Use the calculate function and explain your approach."""
            )

        with conf.thinking(include_thoughts=True, budget=512):
            with conf.tools(tool), conf.tool_config("ANY"):
                response = await generate()
        assert response.candidates

        parts = response.candidates[0].content.parts
        thought_parts = [p for p in parts if p.thought]
        function_parts = [p for p in parts if p.functionCall]
        signature_parts = [p for p in parts if p.thoughtSignature]
        assert thought_parts
        assert function_parts
        assert signature_parts


@test
async def test_thinking_multi_turn() -> None:
    with conf.thinking(include_thoughts=True, budget=512):
        with new_chat():
            with from_user():
                line(
                    "What factors should we consider when designing a sustainable city?"
                )

            response1 = await generate()
            assert response1.candidates

            with from_user():
                line("Based on those factors, what would be the top 3 priorities?")

            response2 = await generate()
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
    spawn(test_thinking_multi_turn)
    # spawn(test_thinking_basic)
    # spawn(test_thinking_dynamic)
    # spawn(test_thinking_with_function_calling)

    # spawn(test_file_upload_and_processing)
    spawn(test_basic_text_generation)
    spawn(test_streaming_response)
    # spawn(test_list_files)
    # spawn(test_function_calling)
    # spawn(test_multi_turn_conversation)
    # spawn(test_invalid_model)
    # spawn(test_thinking_disabled)


if __name__ == "__main__":
    import anyio

    anyio.run(main)
