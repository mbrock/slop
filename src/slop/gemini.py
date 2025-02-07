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
import rich
import logging

logger = logging.getLogger(__name__)


class FunctionParameter(BaseModel):
    type: str
    description: str


class FunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object with type, properties, and required fields",
    )


class FunctionCallingConfig(BaseModel):
    mode: Literal["ANY", "AUTO", "NONE"]
    allowedFunctionNames: Optional[List[str]] = None


class ToolConfig(BaseModel):
    functionCallingConfig: Optional[FunctionCallingConfig] = None


class Tool(BaseModel):
    functionDeclarations: List[FunctionDeclaration]


class Blob(BaseModel):
    mimeType: str
    data: str  # base64-encoded string


class FunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]


class FunctionResponse(BaseModel):
    name: str
    response: Dict[str, Any]


class FileData(BaseModel):
    fileUri: str
    mimeType: Optional[str] = None


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
    output: Optional[str] = None


class Part(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[Blob] = None
    functionCall: Optional[FunctionCall] = None
    functionResponse: Optional[FunctionResponse] = None

    fileData: Optional[FileData] = None
    executableCode: Optional[ExecutableCode] = None
    codeExecutionResult: Optional[CodeExecutionResult] = None


class Content(BaseModel):
    role: str
    parts: List[Part]


class GenerationConfig(BaseModel):
    stopSequences: Optional[List[str]] = None
    responseMimeType: Optional[str] = None
    candidateCount: Optional[int] = Field(default=1, ge=1, le=1)
    maxOutputTokens: Optional[int] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    topP: Optional[float] = None
    topK: Optional[int] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    responseLogprobs: Optional[bool] = None
    logprobs: Optional[int] = None
    enableEnhancedCivicAnswers: Optional[bool] = None


class GenerateRequest(BaseModel):
    contents: Union[Content, List[Content]]
    tools: Optional[List[Tool]] = None
    toolConfig: Optional[ToolConfig] = None
    generationConfig: Optional[GenerationConfig] = None


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
    cachedContentTokenCount: Optional[int] = None
    candidatesTokenCount: int
    totalTokenCount: int


class BlockReason(str, Enum):
    UNSPECIFIED = "BLOCK_REASON_UNSPECIFIED"
    SAFETY = "SAFETY"
    OTHER = "OTHER"
    BLOCKLIST = "BLOCKLIST"
    PROHIBITED_CONTENT = "PROHIBITED_CONTENT"


class PromptFeedback(BaseModel):
    blockReason: Optional[BlockReason] = None
    safetyRatings: List[SafetyRating]


class Candidate(BaseModel):
    content: Content
    finishReason: Optional[FinishReason] = None
    safetyRatings: List[SafetyRating] = []
    tokenCount: Optional[int] = None
    index: Optional[int] = None


class GenerateContentResponse(BaseModel):
    candidates: List[Candidate]
    promptFeedback: Optional[PromptFeedback] = None
    usageMetadata: Optional[UsageMetadata] = None
    modelVersion: Optional[str] = None


class FileState(str, Enum):
    UNSPECIFIED = "STATE_UNSPECIFIED"
    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class Status(BaseModel):
    code: int
    message: str
    details: List[Dict[str, Any]] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    videoDuration: str


class File(BaseModel):
    name: str
    displayName: Optional[str] = None
    mimeType: Optional[str] = None
    sizeBytes: Optional[str] = None
    createTime: Optional[str] = None
    updateTime: Optional[str] = None
    expirationTime: Optional[str] = None
    sha256Hash: Optional[str] = None
    uri: str
    state: Optional[FileState] = None
    error: Optional[Status] = None
    videoMetadata: Optional[VideoMetadata] = None


class FileList(BaseModel):
    files: List[File]
    nextPageToken: Optional[str] = None


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in GOOGLE_API_KEY environment variable"
            )
        self.base_url = "https://generativelanguage.googleapis.com"

    async def generate_content(
        self,
        request: GenerateRequest,
        model: str = "gemini-2.0-flash-exp",
    ) -> AsyncIterator[GenerateContentResponse]:
        url = f"{self.base_url}/v1beta/models/{model}:streamGenerateContent"

        async with httpx.AsyncClient() as client:
            json = request.model_dump(exclude_none=True)
            import rich

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

    async def upload_bytes(
        self,
        data: bytes,
        mime_type: str,
        display_name: Optional[str] = None,
    ) -> File:
        """Upload bytes data to the Gemini API.

        Args:
            data: The bytes to upload
            mime_type: MIME type of the data
            display_name: Optional display name for the file

        Returns:
            File object containing metadata about the uploaded file
        """
        file_size = len(data)

        # Initial resumable upload request
        url = f"{self.base_url}/upload/v1beta/files"
        async with httpx.AsyncClient() as client:
            # Start the upload
            headers = {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            }
            metadata = (
                {"file": {"display_name": display_name}}
                if display_name
                else {"file": {}}
            )
            logger.info(f"Uploading file: {metadata}")
            response = await client.post(
                url,
                params={"key": self.api_key},
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
            response = await client.post(
                upload_url,
                headers=headers,
                content=data,
                timeout=60 * 5,
            )
            response.raise_for_status()
            return File.model_validate(response.json()["file"])

    async def upload_file(
        self,
        file_path: Union[str, Path],
        display_name: Optional[str] = None,
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

        return await self.upload_bytes(data, mime_type, display_name)

    async def get_file(self, file_name: str) -> File:
        """Get metadata for a specific file.

        Args:
            file_name: The name of the file (e.g. 'files/abc-123')

        Returns:
            File object containing metadata about the file
        """
        url = f"{self.base_url}/{file_name}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"key": self.api_key})
            response.raise_for_status()
            return File.model_validate(response.json())

    async def list_files(
        self,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> FileList:
        """List all files owned by the requesting project.

        Args:
            page_size: Maximum number of files to return (default 10, max 100)
            page_token: Page token from a previous list call

        Returns:
            FileList object containing the list of files and next page token
        """
        url = f"{self.base_url}/files"
        params = {"key": self.api_key}
        if page_size:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return FileList.model_validate(response.json())

    async def delete_file(self, file_name: str) -> None:
        """Delete a file.

        Args:
            file_name: The name of the file to delete (e.g. 'files/abc-123')
        """
        url = f"{self.base_url}/v1beta/{file_name}"
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, params={"key": self.api_key})
            response.raise_for_status()

    async def generate_content_sync(
        self,
        request: GenerateRequest,
        #    model: str = "gemini-2.0-flash-exp",
        model: str = "gemini-2.0-pro-exp-02-05",
    ) -> GenerateContentResponse:
        """Non-streaming version of generate_content"""
        url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        async with httpx.AsyncClient() as client:
            rich.print(request.model_dump(exclude_none=True))
            response = await client.post(
                url,
                params={"key": self.api_key},
                json=request.model_dump(exclude_none=True),
                headers={"Content-Type": "application/json"},
                timeout=60 * 5,
            )
            if response.is_error:
                rich.print(response.json())
            response.raise_for_status()
            return GenerateContentResponse.model_validate(response.json())


# Example usage:
async def main():
    client = GeminiClient()

    # Upload the audio file
    file = await client.upload_file(
        "media/interview.ogg", display_name="Interview Audio"
    )
    print(f"Uploaded file: {file.name}")

    # Request transcription
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

    # Generate transcription
    print("\nTranscription:")
    async for response in client.generate_content(request):
        print(response.candidates[0].content.parts[0].text, end="")
    print()

    # Clean up
    await client.delete_file(file.name)


if __name__ == "__main__":
    import trio

    trio.run(main)
