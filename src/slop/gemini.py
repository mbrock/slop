import os
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
)

import httpx
from httpx_sse import ServerSentEvent, aconnect_sse
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic.alias_generators import to_camel


class GoogleBaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)


class FunctionParameter(GoogleBaseModel):
    type: str
    description: str


class FunctionDeclaration(GoogleBaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object with type, properties, and required fields",
    )


class Tool(GoogleBaseModel):
    function_declarations: List[FunctionDeclaration]


class Blob(GoogleBaseModel):
    mime_type: str
    data: str  # base64-encoded string


class FunctionCall(GoogleBaseModel):
    name: str
    args: Dict[str, Any]


class FunctionResponse(GoogleBaseModel):
    name: str
    response: Dict[str, Any]


class FileData(GoogleBaseModel):
    mime_type: Optional[str] = None
    file_uri: str


class Language(str, Enum):
    UNSPECIFIED = "LANGUAGE_UNSPECIFIED"
    PYTHON = "PYTHON"


class ExecutableCode(GoogleBaseModel):
    language: Language
    code: str


class Outcome(str, Enum):
    UNSPECIFIED = "OUTCOME_UNSPECIFIED"
    OK = "OUTCOME_OK"
    FAILED = "OUTCOME_FAILED"
    DEADLINE_EXCEEDED = "OUTCOME_DEADLINE_EXCEEDED"


class CodeExecutionResult(GoogleBaseModel):
    outcome: Outcome
    output: Optional[str] = None


class TextPart(GoogleBaseModel):
    text: str


class InlineDataPart(GoogleBaseModel):
    inline_data: Optional[Blob] = None


class FunctionCallPart(GoogleBaseModel):
    function_call: Optional[FunctionCall] = None


class FunctionResponsePart(GoogleBaseModel):
    function_response: Optional[FunctionResponse] = None


class FileDataPart(GoogleBaseModel):
    file_data: Optional[FileData] = None
    executable_code: Optional[ExecutableCode] = None
    code_execution_result: Optional[CodeExecutionResult] = None


Part = (
    TextPart | InlineDataPart | FunctionCallPart | FunctionResponsePart | FileDataPart
)


class Content(GoogleBaseModel):
    role: str
    parts: List[Part]


class GenerateRequest(GoogleBaseModel):
    contents: Union[Content, List[Content]]
    tools: Optional[List[Tool]] = None


# class NotNullable:
#     def __get_pydantic_core_schema__(
#         self, source: Type[Any], handler: GetCoreSchemaHandler
#     ):
#         schema = handler(source)
#         assert schema["type"] == "nullable"
#         cs = schema["schema"]
#         import rich

#         rich.inspect(schema)
#         return schema


# T = TypeVar("T")
# Omissible = Annotated[Optional[T], NotNullable()]


class MovieSearchParams(BaseModel):
    """Parameters for searching movies in theaters"""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    description: str = Field(..., description="Movie description or genre")

    class Config:
        @staticmethod
        def json_schema_extra(
            schema: Dict[str, Any], model: Type["MovieSearchParams"]
        ) -> None:
            schema.pop("title", None)
            for prop in schema.get("properties", {}).values():
                prop.pop("title", None)


class HarmCategory(str, Enum):
    UNSPECIFIED = "HARM_CATEGORY_UNSPECIFIED"
    HARASSMENT = "HARM_CATEGORY_HARASSMENT"
    HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
    SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
    CIVIC_INTEGRITY = "HARM_CATEGORY_CIVIC_INTEGRITY"


class HarmProbability(str, Enum):
    UNSPECIFIED = "HARM_PROBABILITY_UNSPECIFIED"
    NEGLIGIBLE = "NEGLIGIBLE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class HarmBlockThreshold(str, Enum):
    UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"
    BLOCK_NONE = "BLOCK_NONE"
    OFF = "OFF"


class SafetyRating(GoogleBaseModel):
    category: HarmCategory
    probability: HarmProbability
    blocked: bool = False


class SafetySetting(GoogleBaseModel):
    category: HarmCategory
    threshold: HarmBlockThreshold


class FinishReason(str, Enum):
    UNSPECIFIED = "FINISH_REASON_UNSPECIFIED"
    STOP = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY = "SAFETY"
    RECITATION = "RECITATION"
    LANGUAGE = "LANGUAGE"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"
    SPII = "SPII"
    MALFORMED_FUNCTION_CALL = "MALFORMED_FUNCTION_CALL"


class UsageMetadata(GoogleBaseModel):
    prompt_token_count: int
    cached_content_token_count: Optional[int] = None
    candidates_token_count: int
    total_token_count: int


class BlockReason(str, Enum):
    UNSPECIFIED = "BLOCK_REASON_UNSPECIFIED"
    SAFETY = "SAFETY"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"


class PromptFeedback(GoogleBaseModel):
    block_reason: Optional[BlockReason] = None
    safety_ratings: List[SafetyRating]


class Candidate(GoogleBaseModel):
    content: Content
    finish_reason: Optional[FinishReason] = None
    safety_ratings: List[SafetyRating] = []
    token_count: Optional[int] = None
    index: Optional[int] = None


class GenerateContentResponse(GoogleBaseModel):
    candidates: List[Candidate]
    prompt_feedback: Optional[PromptFeedback] = None
    usage_metadata: Optional[UsageMetadata] = None
    model_version: Optional[str] = None


class GenerationConfig(GoogleBaseModel):
    stop_sequences: Optional[List[str]] = None
    response_mime_type: Optional[str] = None
    candidate_count: Optional[int] = Field(default=1, ge=1, le=1)
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    response_logprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    enable_enhanced_civic_answers: Optional[bool] = None


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in GOOGLE_API_KEY environment variable"
            )
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate_content(
        self,
        request: GenerateRequest,
        model: str = "gemini-2.0-flash-exp",
    ) -> AsyncIterator[GenerateContentResponse]:
        url = f"{self.base_url}/models/{model}:streamGenerateContent"

        async with httpx.AsyncClient() as client:
            json = request.model_dump(exclude_none=True)
            import rich

            rich.print(json)
            async with aconnect_sse(
                client,
                "POST",
                url,
                params={"key": self.api_key, "alt": "sse"},
                json=json,
            ) as sse:
                async for event in sse.aiter_sse():
                    match event:
                        case ServerSentEvent(event="message", data=json):
                            yield GenerateContentResponse.model_validate_json(json)
                        case _:
                            rich.print(event)

            # response = await client.post(
            #     url,
            #     params={"key": self.api_key},
            #     json=json,
            #     headers={"Content-Type": "application/json"},
            # )
            # body = response.json()
            # if response.is_error or isinstance(body, list):
            #     rich.print(body)
            # response.raise_for_status()
            # return GenerateContentResponse(**body)


# Example usage:
async def main():
    client = GeminiClient()

    # Create a function declaration using the Pydantic model's schema
    find_movies = FunctionDeclaration(
        name="find_movies",
        description="Find movie titles currently playing in theaters",
        parameters=MovieSearchParams.model_json_schema(),
    )

    # Create a request
    request = GenerateRequest(
        contents=[
            Content(
                role="user",
                parts=[
                    TextPart(text="What comedy movies are playing in Mountain View?")
                ],
            )
        ],
        tools=[Tool(functionDeclarations=[find_movies])],
    )

    # Generate content
    async for response in client.generate_content(request):
        import rich

        rich.print(response)


if __name__ == "__main__":
    import trio

    trio.run(main)
