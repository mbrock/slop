import hashlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    NotRequired,
    TypedDict,
)

import httpx
import rich
from httpx_sse import ServerSentEvent, aconnect_sse
from pydantic import BaseModel, Field

from slop.parameter import Parameter

logger = logging.getLogger(__name__)

base_url = "https://generativelanguage.googleapis.com"

http_client = Parameter[httpx.AsyncClient]("gemini_http_client")
model = Parameter[str]("gemini_model")
api_key = Parameter[str]("gemini_api_key")
uploader = Parameter[Callable[[bytes, str], Awaitable["File"]] | None](
    "gemini_uploader"
)
generation_config = Parameter["GenerationConfig | None"]("gemini_generation_config")
tools = Parameter["list[Tool] | None"]("gemini_tools")
tool_config = Parameter["ToolConfig | None"]("gemini_tool_config")


class FunctionParameter(TypedDict):
    type: str
    description: str


class FunctionDeclaration(TypedDict):
    name: str
    description: str
    parameters: NotRequired[
        dict[str, Any]
    ]  # JSON Schema object with type, properties, and required fields


class FunctionCallingConfig(TypedDict):
    mode: Literal["ANY", "AUTO", "NONE"]
    allowedFunctionNames: NotRequired[list[str] | None]


class ToolConfig(TypedDict, total=False):
    functionCallingConfig: FunctionCallingConfig | None


class Tool(TypedDict):
    functionDeclarations: list[FunctionDeclaration]


class Blob(BaseModel):
    mimeType: str
    data: str  # base64-encoded string


class FunctionCall(BaseModel):
    name: str
    args: dict[str, Any]


class FunctionResponse(BaseModel):
    name: str
    response: dict[str, Any]


class FileData(BaseModel):
    fileUri: str
    mimeType: str | None = None


class Language(str, Enum):
    UNSPECIFIED = "LANGUAGE_UNSPECIFIED"
    PYTHON = "PYTHON"


class ExecutableCode(BaseModel):
    language: Language
    code: str


class Outcome(str, Enum):
    UNSPECIFIED = "OUTCOME_UNSPECIFIED"
    OK = "OUTCOME_OK"
    FAILED = "OUTCOME_FAILED"
    DEADLINE_EXCEEDED = "OUTCOME_DEADLINE_EXCEEDED"


class CodeExecutionResult(BaseModel):
    outcome: Outcome
    output: str | None = None


class ThinkingConfig(TypedDict, total=False):
    """Configuration for model thinking/reasoning.

    Thinking allows models to process complex requests with internal reasoning.
    - includeThoughts: If true, thought summaries are returned when available
    - thinkingBudget: Number of thought tokens to generate
      * 0: Disable thinking
      * -1: Dynamic thinking (model decides based on complexity)
      * >0: Fixed budget of tokens
    """

    includeThoughts: bool | None  # Include thought summaries in response when available
    thinkingBudget: int | None  # Thought token budget: 0=disabled, -1=dynamic, >0=fixed


class Part(BaseModel):
    text: str | None = None
    inlineData: Blob | None = None
    functionCall: FunctionCall | None = None
    functionResponse: FunctionResponse | None = None
    fileData: FileData | None = None
    executableCode: ExecutableCode | None = None
    codeExecutionResult: CodeExecutionResult | None = None
    thought: bool | None = Field(
        default=None, description="Indicates if this part is a thought from the model"
    )
    thoughtSignature: str | None = Field(
        default=None,
        description="Opaque signature for thought context in multi-turn conversations",
    )


class Content(BaseModel):
    role: str
    parts: list[Part] = Field(default_factory=list)


class GenerationConfig(TypedDict, total=False):
    stopSequences: list[str] | None
    responseMimeType: str | None
    candidateCount: int | None  # default=1, must be 1
    maxOutputTokens: int | None
    temperature: float | None  # range: 0.0 to 2.0
    topP: float | None
    topK: int | None
    presencePenalty: float | None
    frequencyPenalty: float | None
    responseLogprobs: bool | None
    logprobs: int | None
    enableEnhancedCivicAnswers: bool | None
    thinkingConfig: (
        ThinkingConfig | None
    )  # Configuration for model thinking/reasoning (Gemini 2.5+)


class GenerateRequest(BaseModel):
    contents: Content | list[Content]
    tools: list[Tool] | None = None
    toolConfig: ToolConfig | None = None
    generationConfig: GenerationConfig | None = None


class MovieSearchParams(BaseModel):
    """Parameters for searching movies in theaters"""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    description: str = Field(..., description="Movie description or genre")

    model_config = {
        "json_schema_extra": lambda schema, model: (
            schema.pop("title", None),
            [prop.pop("title", None) for prop in schema.get("properties", {}).values()],
        )[-1]
    }


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


class SafetyRating(BaseModel):
    category: HarmCategory
    probability: HarmProbability
    blocked: bool = False


class SafetySetting(BaseModel):
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


class UsageMetadata(BaseModel):
    promptTokenCount: int
    cachedContentTokenCount: int | None = None
    candidatesTokenCount: int | None = None
    totalTokenCount: int


class BlockReason(str, Enum):
    UNSPECIFIED = "BLOCK_REASON_UNSPECIFIED"
    SAFETY = "SAFETY"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"


class PromptFeedback(BaseModel):
    blockReason: BlockReason | None = None
    safetyRatings: list[SafetyRating] | None = None


class Candidate(BaseModel):
    content: Content
    finishReason: FinishReason | None = None
    safetyRatings: list[SafetyRating] = []
    tokenCount: int | None = None
    index: int | None = None


class GenerateContentResponse(BaseModel):
    candidates: list[Candidate]
    promptFeedback: PromptFeedback | None = None
    usageMetadata: UsageMetadata | None = None
    modelVersion: str | None = None


class FileState(str, Enum):
    UNSPECIFIED = "STATE_UNSPECIFIED"
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class Status(BaseModel):
    code: int
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    videoDuration: str


class File(BaseModel):
    name: str
    displayName: str | None = None
    mimeType: str | None = None
    sizeBytes: str | None = None
    createTime: str | None = None
    updateTime: str | None = None
    expirationTime: str | None = None
    sha256Hash: str | None = None
    uri: str
    state: FileState | None = None
    error: Status | None = None
    videoMetadata: VideoMetadata | None = None


class FileList(BaseModel):
    files: list[File]
    nextPageToken: str | None = None


class GeminiError(Exception):
    """Base class for Gemini API errors."""

    def __init__(self, message: str, error_data: dict | None = None):
        self.message = message
        self.error_data = error_data
        super().__init__(message)


class ModelOverloadedError(GeminiError):
    """Raised when the model is overloaded and we could try an alternative."""

    def __init__(self, current_model: str):
        self.current_model = current_model
        self.alternative_model = (
            "gemini-2.5-flash" if "pro" in current_model else "gemini-2.5-flash-lite"
        )
        super().__init__(
            f"Model {current_model} is overloaded. Consider trying {self.alternative_model}"
        )


async def generate_content(
    request: GenerateRequest,
) -> AsyncIterator[GenerateContentResponse]:
    model_to_use = model.get()
    url = f"{base_url}/v1beta/models/{model_to_use}:streamGenerateContent"

    json = request.model_dump(exclude_none=True)

    async with aconnect_sse(
        http_client.get(),
        "POST",
        url,
        params={"key": api_key.get(), "alt": "sse"},
        json=json,
    ) as sse:
        async for event in sse.aiter_sse():
            match event:
                case ServerSentEvent(event="message", data=json):
                    yield GenerateContentResponse.model_validate_json(json)
                case _:
                    rich.print(event)


async def get_file_metadata(file_name: str) -> File | None:
    """Get metadata for a specific file.

    Args:
        file_name: The name of the file (e.g. 'files/abc-123')

    Returns:
        File object containing metadata about the file, or None if not found
    """
    url = f"{base_url}/v1beta/{file_name}"
    response = await http_client.get().get(url, params={"key": api_key.get()})
    if response.status_code == 404 or response.status_code == 403:
        return None
    response.raise_for_status()
    return File.model_validate(response.json())


def _generate_content_addressed_name(data: bytes, mime_type: str) -> str:
    """Generate a content-addressed file name based on data hash."""
    hash_obj = hashlib.sha256()
    hash_obj.update(data)
    hash_obj.update(mime_type.encode())
    return f"files/{hash_obj.hexdigest()[:16]}"


async def upload(
    data: bytes,
    mime_type: str,
    display_name: str | None = None,
) -> File:
    """Upload bytes data to the Gemini API.

    Args:
        data: The bytes to upload
        mime_type: MIME type of the data
        display_name: Optional display name for the file

    Returns:
        File object containing metadata about the uploaded file
    """
    # Generate content-addressed name
    file_name = _generate_content_addressed_name(data, mime_type)

    # Check if file already exists
    if existing_file := await get_file_metadata(file_name):
        if existing_file.state == FileState.ACTIVE:
            logger.info(f"File already exists: {file_name}")
            return existing_file
        else:
            logger.info(f"File exists but not active, state: {existing_file.state}")

    file_size = len(data)

    # Initial resumable upload request
    url = f"{base_url}/upload/v1beta/files"
    # Start the upload
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json",
    }
    metadata = {
        "file": {
            "name": file_name,
            **({"display_name": display_name} if display_name else {}),
        }
    }
    logger.info(f"Uploading file: {metadata}")
    response = await http_client.get().post(
        url,
        params={"key": api_key.get()},
        headers=headers,
        json=metadata,
        timeout=60 * 5,
    )
    response.raise_for_status()

    # Get upload URL from response headers
    upload_url = response.headers.get("x-goog-upload-url")
    if not upload_url:
        raise ValueError("No upload URL received from server")

    # Upload the data
    headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    response = await http_client.get().post(
        upload_url,
        headers=headers,
        content=data,
        timeout=60 * 5,
    )
    response.raise_for_status()
    return File.model_validate(response.json()["file"])


async def upload_file(
    file_path: str | Path,
    display_name: str | None = None,
) -> File:
    """Upload a file to the Gemini API.

    Args:
        file_path: Path to the file to upload
        display_name: Optional display name for the file

    Returns:
        File object containing metadata about the uploaded file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file info
    mime_type = os.popen(f"file -b --mime-type {file_path}").read().strip()

    # Read file data
    with open(file_path, "rb") as f:
        data = f.read()

    return await upload(data, mime_type, display_name)


async def resolve_blob_uris(contents: list[Content]) -> None:
    """Upload any blob: URIs in contents and update them with real URIs.

    Modifies contents in place. Requires the uploader parameter to be set.
    """
    upload_fn = uploader.peek()
    if not upload_fn:
        # No uploader configured, just return
        return

    import anyio
    from slop.models import get_blob

    # Collect all blob URIs and their parts
    blobs: dict[str, list[Part]] = {}

    for content in contents:
        for part in content.parts:
            if not part.fileData or not part.fileData.fileUri:
                continue
            uri = part.fileData.fileUri
            if uri.startswith("blob:"):
                blob_hash = uri[5:]  # Skip "blob:" prefix
                blobs.setdefault(blob_hash, []).append(part)

    if not blobs:
        return

    # Upload all blobs in parallel
    uploads: dict[str, File] = {}

    async def upload_blob(blob_hash: str) -> None:
        data, mime = get_blob(blob_hash)
        # Use the first part's MIME type if specified, otherwise use stored
        first_part = blobs[blob_hash][0]
        if first_part.fileData and first_part.fileData.mimeType:
            mime = first_part.fileData.mimeType
        uploads[blob_hash] = await upload_fn(data, mime)

    async with anyio.create_task_group() as tg:
        for blob_hash in blobs:
            tg.start_soon(upload_blob, blob_hash)

    # Update all parts with uploaded URIs
    for blob_hash, parts in blobs.items():
        uploaded = uploads[blob_hash]
        for part in parts:
            if part.fileData:
                part.fileData.fileUri = uploaded.uri
                # Keep part's MIME type if specified, otherwise use uploaded
                if not part.fileData.mimeType:
                    part.fileData.mimeType = uploaded.mimeType


async def get_file(file_name: str) -> File:
    """Get metadata for a specific file.

    Args:
        file_name: The name of the file (e.g. 'files/abc-123')

    Returns:
        File object containing metadata about the file
    """
    url = f"{base_url}/{file_name}"
    response = await http_client.get().get(url, params={"key": api_key.get()})
    response.raise_for_status()
    return File.model_validate(response.json())


async def list_files(
    page_size: int | None = None,
    page_token: str | None = None,
) -> FileList:
    """List all files owned by the requesting project.

    Args:
        page_size: Maximum number of files to return (default 10, max 100)
        page_token: Page token from a previous list call

    Returns:
        FileList object containing the list of files and next page token
    """
    url = f"{base_url}/v1beta/files"
    params = {"key": api_key.get()}
    if page_size:
        params["pageSize"] = page_size
    if page_token:
        params["pageToken"] = page_token

    response = await http_client.get().get(url, params=params)
    response.raise_for_status()
    return FileList.model_validate(response.json())


async def delete_file(file_name: str) -> None:
    """Delete a file.

    Args:
        file_name: The name of the file to delete (e.g. 'files/abc-123')
    """
    url = f"{base_url}/v1beta/{file_name}"
    response = await http_client.get().delete(url, params={"key": api_key.get()})
    response.raise_for_status()


async def generate_content_sync(
    request: GenerateRequest,
) -> GenerateContentResponse:
    """Non-streaming version of generate_content"""
    model_to_use = model.get()
    url = f"{base_url}/v1beta/models/{model_to_use}:generateContent"
    response = await http_client.get().post(
        url,
        params={"key": api_key.get()},
        json=request.model_dump(exclude_none=True),
        headers={"Content-Type": "application/json"},
        timeout=60 * 5,
    )
    if response.is_error:
        error_data = response.json()

        error_status = error_data.get("error", {}).get("status")
        error_message = error_data.get("error", {}).get(
            "message", "Unknown error occurred"
        )

        if error_status == "UNAVAILABLE":
            raise ModelOverloadedError(model_to_use)
        elif error_status == "INTERNAL":
            raise GeminiError(
                "Internal server error occurred. Please try again.",
                error_data,
            )
        elif error_status == "INVALID_ARGUMENT":
            raise GeminiError(f"Invalid request: {error_message}", error_data)
        else:
            raise GeminiError(f"API error: {error_message}", error_data)

    response.raise_for_status()

    return GenerateContentResponse.model_validate(response.json())


async def generate(
    contents: list[Content],
) -> GenerateContentResponse:
    """Generate a response (does not modify contents)."""
    final_config = generation_config.peek()
    final_tools = tools.peek()
    final_tool_config = tool_config.peek()

    # Resolve blobs if needed
    contents_copy = [content.model_copy(deep=True) for content in contents]
    await resolve_blob_uris(contents_copy)

    request = GenerateRequest(
        contents=contents_copy,
        generationConfig=final_config,
        tools=final_tools,
        toolConfig=final_tool_config,
    )
    response = await generate_content_sync(request)

    return response


async def generate_streaming(
    contents: list[Content],
) -> AsyncIterator[GenerateContentResponse]:
    """Generate a streaming response, yielding chunks as they arrive."""
    final_config = generation_config.peek()
    final_tools = tools.peek()
    final_tool_config = tool_config.peek()

    # Resolve blobs if needed
    contents_copy = [content.model_copy(deep=True) for content in contents]
    await resolve_blob_uris(contents_copy)

    request = GenerateRequest(
        contents=contents_copy,
        generationConfig=final_config,
        tools=final_tools,
        toolConfig=final_tool_config,
    )

    async for response in generate_content(request):
        yield response


# Example usage:
async def main():
    async with httpx.AsyncClient() as client:
        with (
            http_client.using(client),
            model.using("gemini-2.5-flash-lite"),
            api_key.using_env("GOOGLE_API_KEY"),
        ):
            file = await upload_file(
                "media/interview.ogg", display_name="Interview Audio"
            )
            print(f"Uploaded file: {file.name}")

            request = GenerateRequest(
                contents=[
                    Content(
                        role="user",
                        parts=[
                            Part(
                                text="""Please transcribe this audio file. Use HTML with elements like <span data-speaker="S1 | S2 | ..." data-time="hh:mm:ss"> for each utterance. Use dashes (—), ellipses (…), and light editing for an accurate transcript with typographic care. Write disfluencies like "it's— well— you know—"."""
                            ),
                            Part(fileData=FileData(fileUri=file.uri)),
                        ],
                    )
                ],
            )

            print("\nTranscription:")
            async for response in generate_content(request):
                print(response.candidates[0].content.parts[0].text, end="")
            print()

            # Clean up
            await delete_file(file.name)


if __name__ == "__main__":
    import anyio

    anyio.run(main)
