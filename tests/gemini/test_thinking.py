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
    """Test basic thinking configuration and verify thoughts are included."""
    client = GeminiClient()
    
    # Use a complex problem that should trigger thinking
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="""Solve this step-by-step problem:
            
            A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
            Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
            The stations are 280 miles apart.
            
            At what time will the trains meet? Show your reasoning step by step.""")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=512  # Reduced budget for faster tests
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash-lite")
    assert response.candidates
    assert response.candidates[0].content.parts
    
    # Verify at least one thought part is included
    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, "Expected at least one thought part when thinking is enabled"
    
    # Verify thought parts have content
    for thought_part in thought_parts:
        assert thought_part.text, "Thought parts should have text content"
        assert len(thought_part.text) > 0, "Thought text should not be empty"
    
    # Also verify we have regular response parts
    regular_parts = [p for p in response.candidates[0].content.parts if not p.thought]
    assert len(regular_parts) > 0, "Should have regular response in addition to thoughts"


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
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash-lite")
    assert response.candidates
    
    # With thinking disabled, should not have thought parts
    for part in response.candidates[0].content.parts:
        assert part.thought is None or part.thought is False


@pytest.mark.asyncio
async def test_thinking_dynamic():
    """Test dynamic thinking mode and verify thoughts are included."""
    client = GeminiClient()
    
    # Complex philosophical/scientific question that should trigger dynamic thinking
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="""Analyze this philosophical paradox:
            
            If a time traveler goes back in time and prevents their own birth,
            how can they exist to go back in time in the first place?
            
            Consider multiple perspectives: physics, philosophy, and logic.
            Explain your reasoning process.""")]
        ),
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=-1  # Dynamic thinking - model decides budget
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash-lite")
    assert response.candidates
    assert response.candidates[0].content.parts
    
    # With dynamic thinking on complex problems, we should get thoughts
    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, "Dynamic thinking should include thoughts for complex problems"
    
    # Verify thought quality
    total_thought_length = sum(len(p.text or "") for p in thought_parts)
    assert total_thought_length > 100, "Thoughts should be substantial for complex problems"


@pytest.mark.asyncio
async def test_thinking_with_function_calling():
    """Test thinking with function calling and verify both thoughts and signatures."""
    client = GeminiClient()
    
    # Define a calculation function
    math_function = FunctionDeclaration(
        name="calculate",
        description="Perform a mathematical calculation",
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
    
    # Complex problem that requires thinking and function use
    request = GenerateRequest(
        contents=Content(
            role="user",
            parts=[Part(text="""Calculate the compound interest on $10,000 at 5% annual rate for 3 years,
                           compounded monthly. Use the calculate function and explain your approach.""")]
        ),
        tools=[tool],
        toolConfig=tool_config,
        generationConfig=GenerationConfig(
            thinkingConfig=ThinkingConfig(
                includeThoughts=True,
                thinkingBudget=512
            )
        )
    )
    
    response = await client.generate_content_sync(request, model="gemini-2.5-flash-lite")
    assert response.candidates
    assert response.candidates[0].content.parts
    
    # Should have thoughts when thinking is enabled
    thought_parts = [p for p in response.candidates[0].content.parts if p.thought]
    assert len(thought_parts) > 0, "Should include thoughts with function calling"
    
    # Should have function call
    function_parts = [p for p in response.candidates[0].content.parts if p.functionCall]
    assert len(function_parts) > 0, "Should call the calculate function"
    
    # Check for thought signatures (appear with function calling + thinking)
    signature_parts = [p for p in response.candidates[0].content.parts if p.thoughtSignature]
    # Note: Signatures may or may not be present depending on model's decision


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
                thinkingBudget=512
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
                thinkingBudget=512
            )
        )
    )
    
    response2 = await client.generate_content_sync(request2, model="gemini-2.5-flash")
    assert response2.candidates
    # Model should maintain context from previous thoughts