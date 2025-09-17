"""Test thinking/reasoning features with Gemini 2.5 models."""

import pytest
from src.slop.gemini import (
    GeminiClient,
    GenerateRequest,
    Content,
    Part,
    GenerationConfig,
    ThinkingConfig,
    Tool,
    FunctionDeclaration,
    ToolConfig,
    FunctionCallingConfig,
)


@pytest.mark.asyncio
async def test_thinking_basic():
    """Test basic thinking configuration."""
    client = GeminiClient()
    
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="Solve this step by step: What is 47 * 89 + 234?")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=1000  # Fixed budget
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash")
    assert response.candidates
    
    # Check if any parts contain thoughts
    for part in response.candidates[0].content.parts:
        if part.thought:
            assert part.text  # Thought should have text content
    
    # Note: Model may or may not include thoughts depending on the query


@pytest.mark.asyncio
async def test_thinking_disabled():
    """Test with thinking disabled."""
    client = GeminiClient()
    
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="What is 2 + 2?")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=False,
                thinkingBudget=0  # Disable thinking
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash")
    assert response.candidates
    
    # With thinking disabled, should not have thought parts
    for part in response.candidates[0].content.parts:
        assert part.thought is None or part.thought is False


@pytest.mark.asyncio
async def test_thinking_dynamic():
    """Test dynamic thinking mode."""
    client = GeminiClient()
    
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="Explain the theory of relativity and its implications for time travel.")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=-1  # Dynamic thinking
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash")
    assert response.candidates
    assert response.candidates[0].content.parts


@pytest.mark.asyncio
async def test_thinking_with_function_calling():
    """Test thinking with function calling to get thought signatures."""
    client = GeminiClient()
    
    # Define a simple math function
    math_function = FunctionDeclaration(
        name="calculate",
        description="Perform a calculation",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
    
    tool = Tool(functionDeclarations=[math_function])
    tool_config = ToolConfig(
        functionCallingConfig=FunctionCallingConfig(mode="AUTO")
    )
    
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="Use the calculate function to find the result of 123 * 456")]
        ),
        tools=[tool],
        toolConfig=tool_config,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=500
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash")
    assert response.candidates
    
    # Check for thought signatures when function calling is enabled
    for part in response.candidates[0].content.parts:
        if part.thoughtSignature:
            assert isinstance(part.thoughtSignature, str)
    
    # Note: Signatures only appear when both thinking and function calling are used


@pytest.mark.asyncio
async def test_thinking_multi_turn_with_signatures():
    """Test maintaining thought context across turns using signatures."""
    client = GeminiClient()
    
    # First turn with thinking
    request1 = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="Let's solve a complex problem. First, what factors should we consider when designing a sustainable city?")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=1000
            )
        )
    )
    
    response1 = await client.generate_content_sync(request1, model="gemini-2.5-flash")
    assert response1.candidates
    
    # Build conversation with response including any thought signatures
    conversation = [
        Content(role="user", parts=[Part(text="Let's solve a complex problem. First, what factors should we consider when designing a sustainable city?")]),
        response1.candidates[0].content  # Include full response with signatures
    ]
    
    # Second turn - model should maintain context via signatures
    conversation.append(
        Content(role="user", parts=[Part(text="Based on those factors, what would be the top 3 priorities?")])
    )
    
    request2 = GenerateRequest(
        contents=conversation,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=1000
            )
        )
    )
    
    response2 = await client.generate_content_sync(request2, model="gemini-2.5-flash")
    assert response2.candidates
    # Model should maintain context from previous thoughts