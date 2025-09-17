"""Test thinking/reasoning features with Gemini 2.5 models."""

import pytest

from src.slop.gemini import (
    Content,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiClient,
    GenerateRequest,
    GenerationConfig,
    Part,
    ThinkingConfig,
    Tool,
    ToolConfig,
)


@pytest.mark.asyncio
async def test_thinking_basic(gemini_client: GeminiClient):
    """Test basic thinking configuration and verify thoughts are included."""

    # Use a complex problem that should trigger thinking
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
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=512,
            )
        ),
    )

    response = await gemini_client.generate_content_sync(request)
    assert response.candidates
    assert response.candidates[0].content.parts

    # Verify at least one thought part is included
    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, (
        "Expected at least one thought part when thinking is enabled"
    )

    # Verify thought parts have content
    for thought_part in thought_parts:
        assert thought_part.text, "Thought parts should have text content"
        assert len(thought_part.text) > 0, "Thought text should not be empty"

    # Also verify we have regular response parts
    regular_parts = [p for p in response.candidates[0].content.parts if not p.thought]
    assert len(regular_parts) > 0, (
        "Should have regular response in addition to thoughts"
    )


@pytest.mark.asyncio
async def test_thinking_disabled(gemini_client: GeminiClient):
    """Test with thinking disabled."""

    request = GenerateRequest(
        contents=Content(role="user", parts=[Part(text="What is 2 + 2?")]),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=False,
                thinkingBudget=0,  # Disable thinking
            )
        ),
    )

    response = await gemini_client.generate_content_sync(request)
    assert response.candidates

    for part in response.candidates[0].content.parts:
        assert part.thought is None or part.thought is False


@pytest.mark.asyncio
@pytest.mark.slow
async def test_thinking_dynamic(gemini_client: GeminiClient):
    """Test dynamic thinking mode and verify thoughts are included."""

    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[
                Part(
                    text="Come up with a mildly interesting fact. Think, but not too hard."
                )
            ],
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=-1,
            )
        ),
    )

    response = await gemini_client.generate_content_sync(request)
    assert response.candidates
    assert response.candidates[0].content.parts

    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, "Dynamic thinking should include thoughts"


@pytest.mark.asyncio
async def test_thinking_with_function_calling(gemini_client: GeminiClient):
    """Test thinking with function calling and verify both thoughts and signatures."""

    # Define a calculation function
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

    response = await gemini_client.generate_content_sync(request)
    assert response.candidates
    assert response.candidates[0].content.parts

    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, "Should include thoughts with function calling"

    function_parts = [p for p in response.candidates[0].content.parts if p.functionCall]
    assert len(function_parts) > 0, "Should call the calculate function"

    signature_parts = [
        p for p in response.candidates[0].content.parts if p.thoughtSignature
    ]

    assert len(signature_parts) > 0, "Should include thought signatures"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_thinking_multi_turn_with_signatures(gemini_client: GeminiClient):
    """Test maintaining thought context across turns using signatures."""

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

    response1 = await gemini_client.generate_content_sync(request1)
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
        response1.candidates[0].content,  # Include full response with signatures
    ]

    conversation.append(
        Content(
            role="user",
            parts=[
                Part(text="Based on those factors, what would be the top 3 priorities?")
            ],
        )
    )

    request2 = GenerateRequest(
        contents=conversation,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(includeThoughts=True, thinkingBudget=512)
        ),
    )

    response2 = await gemini_client.generate_content_sync(request2)
    assert response2.candidates
