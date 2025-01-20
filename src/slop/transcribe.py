from contextlib import asynccontextmanager, contextmanager
import hashlib
import json
import logging
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import BinaryIO

from fastapi import FastAPI, UploadFile, HTTPException, Request, Response
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
import rich
from tagflow import (
    DocumentMiddleware,
    Live,
    TagResponse,
    classes,
    tag,
    text,
    attr,
)
import trio

from slop.gemini import (
    Content,
    FileData,
    FunctionCallingConfig,
    FunctionDeclaration,
    GenerateRequest,
    GeminiClient,
    Part,
    Tool,
    ToolConfig,
)
from slop.views import upload_area

logger = logging.getLogger("slop.transcribe")

# Initialize Live instance for WebSocket support
live = Live()


class PartitionSegment(BaseModel):
    """A segment of the interview with start and end times"""

    start_time: str = Field(description="Start time of the segment (HH:MM:SS)")
    end_time: str = Field(description="End time of the segment (HH:MM:SS)")
    phrases: list[str] = Field(
        description="Some example short phrases from this segment to help orient the user"
    )
    audio_hash: str | None = None  # Hash of the segment's audio


class Utterance(BaseModel):
    """A single utterance in the interview"""

    speaker: str = Field(description="Speaker identifier (e.g. 'S1', 'S2')")
    timestamp: str = Field(description="Timestamp of the utterance (HH:MM:SS)")
    text: str = Field(description="The transcribed text")
    audio_hash: str | None = None  # Hash of the utterance's audio segment


class Segment(BaseModel):
    """A segment of the interview containing utterances"""

    start_time: str = Field(description="Start time of the segment (HH:MM:SS)")
    end_time: str = Field(description="End time of the segment (HH:MM:SS)")
    audio_hash: str | None = None  # Hash of the segment's audio
    utterances: list[Utterance] = []


class Interview(BaseModel):
    """An interview with its metadata and transcribed segments"""

    id: str
    filename: str
    file_uri: str
    audio_hash: str | None = None  # Hash of the processed audio file
    current_position: str = Field(
        default="00:00:00", description="Current position in the interview (HH:MM:SS)"
    )
    segments: list[Segment] = []


class Store:
    """A simple key-value store using SQLite and Pydantic"""

    def __init__(self, db_path: str | Path, model_class: type[BaseModel]):
        self.db_path = Path(db_path)
        self.model_class = model_class
        self._init_db()

    def _init_db(self):
        """Initialize the database with a key-value table"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def get(self, key: str) -> BaseModel | None:
        """Get a value by key"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store WHERE key = ?", (key,))
            if row := cursor.fetchone():
                return self.model_class.model_validate_json(row[0])
            return None

    def put(self, key: str, value: BaseModel):
        """Store a value by key"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
                (key, value.model_dump_json()),
            )

    def values(self) -> list[BaseModel]:
        """Get all values"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store")
            return [
                self.model_class.model_validate_json(row[0])
                for row in cursor.fetchall()
            ]


class BlobStore:
    """Content-addressed store for binary data"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database with a blobs table"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    mime_type TEXT NOT NULL
                )
            """)

    def put(self, data: bytes, mime_type: str) -> str:
        """Store binary data and return its hash"""
        hash = hashlib.sha256(data).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blobs (hash, data, mime_type) VALUES (?, ?, ?)",
                (hash, data, mime_type),
            )
        return hash

    def get(self, hash: str) -> tuple[bytes, str] | None:
        """Get binary data and mime type by hash"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data, mime_type FROM blobs WHERE hash = ?", (hash,)
            )
            if row := cursor.fetchone():
                return row[0], row[1]
            return None


# Initialize the stores
INTERVIEWS = Store(Path("data/interviews.db"), Interview)
BLOBS = BlobStore(Path("data/blobs.db"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with live.run(app):
        logger.info("live server started")
        yield
    logger.info("live server stopped")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=TagResponse,
    title="Slop Interview",
)
app.add_middleware(DocumentMiddleware)


@contextmanager
def layout(title: str):
    """Common layout wrapper for all pages"""
    with tag.html(lang="en"):
        with tag.head():
            with tag.title():
                text(f"{title} - Slop")
            with tag.script(src="https://cdn.tailwindcss.com"):
                pass
            with tag.script(src="https://unpkg.com/htmx.org@1.9.10"):
                pass
            with tag.script():
                text("""htmx.config = { "globalViewTransitions": true }""")
            # set tailwind serif font to equity ot
            with tag.script():
                text("""
                    tailwind.config = {
                      theme: {
                        fontFamily: {
                            serif: ["Equity OT", "Times New Roman", "serif"],
                        },
                      },
                    }
                """)
            live.script_tag()

        with tag.body(classes="bg-stone-300 min-h-screen font-serif p-4"):
            yield


@app.get("/")
async def home():
    with layout("Home"):
        with tag.div(classes="prose mx-auto"):
            upload_area()


async def process_audio(input_path: Path) -> bytes:
    """Convert audio to Ogg Vorbis format using ffmpeg"""
    process = await trio.run_process(
        [
            "ffmpeg",
            "-i",
            str(input_path),  # Input file
            "-c:a",
            "libvorbis",  # Vorbis codec
            "-q:a",
            "4",  # Quality level (0-10, 4 is good)
            "-f",
            "ogg",  # Output format
            "pipe:1",  # Output to stdout
        ],
        capture_stdout=True,
        capture_stderr=True,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    return process.stdout


@app.post("/upload")
async def upload_audio(audio: UploadFile):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(audio.filename).suffix
    ) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = Path(tmp.name)

        try:
            # Process and store the audio
            processed_audio = await process_audio(tmp_path)
            audio_hash = BLOBS.put(processed_audio, "audio/ogg")

            # Upload to Gemini
            client = GeminiClient()
            file = await client.upload_file(str(tmp_path), display_name=audio.filename)

            # Create interview record
            interview_id = str(len(list(INTERVIEWS.values())) + 1)
            interview = Interview(
                id=interview_id,
                filename=audio.filename,
                file_uri=file.uri,
                audio_hash=audio_hash,
            )
            INTERVIEWS.put(interview_id, interview)

        finally:
            # Clean up temp file
            tmp_path.unlink()

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


@app.get("/interview/{interview_id}")
async def view_interview(interview_id: str):
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    with layout(f"Interview - {interview.filename}"):
        with tag.div(classes="prose mx-auto"):
            with tag.h1(classes="mb-2 text-bold"):
                text(" ".join(interview.filename.split(".")[:-1]))

            if interview.audio_hash:
                with tag.div(classes="mb-2"):
                    with tag.audio(
                        src=f"/audio/{interview.audio_hash}",
                        controls=True,
                        classes="w-full",
                        preload="metadata",
                    ):
                        pass

            with tag.form(
                action=f"/interview/{interview_id}/transcribe-next",
                method="post",
                classes="mb-4",
            ):
                with tag.button(
                    type="submit",
                    classes="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600",
                ):
                    text("Transcribe Next Segment")

            if interview.segments:
                with tag.div(classes="flex flex-col gap-8"):
                    for segment in interview.segments:
                        with tag.div(
                            classes="flex flex-col gap-2 bg-stone-100 p-4 rounded-lg"
                        ):
                            with tag.div(classes="flex items-center gap-2"):
                                if segment.audio_hash:
                                    with tag.audio(
                                        src=f"/audio/{segment.audio_hash}",
                                        controls=True,
                                        classes="h-8",
                                        preload="metadata",
                                    ):
                                        pass

                            with tag.div(classes="flex flex-wrap gap-4 pl-4"):
                                for utterance in segment.utterances:
                                    with tag.span(
                                        **{"data-speaker": utterance.speaker}
                                    ):
                                        if utterance.speaker == "S1":
                                            classes("font-bold")
                                        text(utterance.text)


PARTITION_TOOL = Tool(
    functionDeclarations=[
        FunctionDeclaration(
            name="partition_interview",
            description="Partition an audio interview into segments of around a minute each",
            parameters={
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {
                                    "type": "string",
                                    "description": "Start time of the segment (HH:MM:SS) (should be 00:00:00 for the first segment)",
                                },
                                "end_time": {
                                    "type": "string",
                                    "description": "End time of the segment (HH:MM:SS) (should be in the middle of a pause or at the end of a speaker turn)",
                                },
                                "phrases": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A brief set of phrases or paraphrases that help identify the segment. It should fit in a line.",
                                    },
                                },
                            },
                            "required": ["start_time", "end_time", "phrases"],
                        },
                        "description": "The segments of the interview",
                    }
                },
                "required": ["segments"],
            },
        )
    ]
)


async def extract_segment(input_path: Path, start_time: str, end_time: str) -> bytes:
    """Extract a segment from an audio file using ffmpeg"""
    rich.print(
        {"input_path": input_path, "start_time": start_time, "end_time": end_time}
    )
    process = await trio.run_process(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ss",
            start_time,  # Start time
            "-to",
            end_time,  # End time
            "-c:a",
            "libvorbis",  # Vorbis codec
            "-q:a",
            "6",  # Quality level
            "-f",
            "ogg",  # Output format
            "pipe:1",  # Output to stdout
        ],
        capture_stdout=True,
        capture_stderr=True,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    return process.stdout


@app.post("/interview/{interview_id}/partition")
async def partition_interview(interview_id: str):
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    # Create a temporary file for the full audio
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        if not interview.audio_hash:
            raise HTTPException(status_code=400, detail="Interview has no audio")

        audio_data, _ = BLOBS.get(interview.audio_hash)
        tmp.write(audio_data)
        tmp.flush()
        tmp_path = Path(tmp.name)

        try:
            client = GeminiClient()
            request = GenerateRequest(
                contents=[
                    Content(
                        role="user",
                        parts=[
                            Part(fileData=FileData(fileUri=interview.file_uri)),
                            Part(
                                text="""Please listen to this interview and partition it into clean segments of ROUGHLY three minutes each, making sure to end each segment in the middle of a natural pause or at the end of a speaker turn.
                                Provide the start and end time for each segment, along with a brief set of phrases that help identify the segment.
                                Use the `partition_interview` tool to provide the segments."""
                            ),
                        ],
                    )
                ],
                tools=[PARTITION_TOOL],
                toolConfig=ToolConfig(
                    functionCallingConfig=FunctionCallingConfig(mode="ANY")
                ),
            )

            response = await client.generate_content_sync(request)
            if not response.candidates:
                raise HTTPException(
                    status_code=500, detail="Failed to partition interview"
                )

            rich.print(response)

            # Look for a function call part among all parts
            function_call = None
            for part in response.candidates[0].content.parts:
                rich.print(part)
                if hasattr(part, "functionCall") and part.functionCall:
                    function_call = part.functionCall
                    break

            if not function_call or function_call.name != "partition_interview":
                raise HTTPException(
                    status_code=500, detail="Invalid response from model"
                )

            # Create segments with audio extracts
            segments = []
            for segment_data in function_call.args["segments"]:
                # Extract audio for this segment
                segment_audio = await extract_segment(
                    tmp_path, segment_data["start_time"], segment_data["end_time"]
                )

                # Store the segment audio
                audio_hash = BLOBS.put(segment_audio, "audio/ogg")

                # Create the segment with audio hash
                segment = Segment(**segment_data, audio_hash=audio_hash)
                segments.append(segment)

            # Update the interview with the new segments
            interview.segments = segments
            INTERVIEWS.put(interview.id, interview)

        finally:
            # Clean up temp file
            tmp_path.unlink()

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


@app.get("/audio/{hash}")
async def get_audio(hash: str, request: Request):
    """Serve audio file by hash with support for range requests"""
    if not (result := BLOBS.get(hash)):
        raise HTTPException(status_code=404, detail="Audio not found")

    data, mime_type = result
    file_size = len(data)

    # Parse range header
    range_header = request.headers.get("range")

    if not range_header:
        # No range requested, return full file
        return Response(
            content=data,
            media_type=mime_type,
            headers={"accept-ranges": "bytes", "content-length": str(file_size)},
        )

    # Parse range header (format: "bytes=start-end")
    try:
        range_str = range_header.replace("bytes=", "")
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid range header")

    # Validate range
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416, detail=f"Range not satisfiable. File size: {file_size}"
        )

    # Calculate content length
    content_length = end - start + 1

    # Return partial content
    return Response(
        content=data[start : end + 1],
        status_code=206,
        media_type=mime_type,
        headers={
            "content-range": f"bytes {start}-{end}/{file_size}",
            "accept-ranges": "bytes",
            "content-length": str(content_length),
        },
    )


def time_to_seconds(time_str: str) -> int:
    """Convert a time string in HH:MM:SS format to seconds"""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def parse_transcription_xml(xml_text: str) -> list[Utterance]:
    """Parse transcription XML into a list of Utterances"""
    import xml.etree.ElementTree as ET
    from io import StringIO

    # Parse XML string
    try:
        tree = ET.parse(StringIO(xml_text))
        root = tree.getroot()

        utterances = []
        for utterance in root.findall("utterance"):
            utterances.append(
                Utterance(
                    speaker=utterance.get("speaker"),
                    timestamp=utterance.get("start"),
                    text=utterance.text.strip() if utterance.text else "",
                )
            )
        return utterances
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        logger.error(f"XML content was: {xml_text}")
        return []


@app.post("/interview/{interview_id}/transcribe-next")
async def transcribe_next_segment(interview_id: str):
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    if not interview.audio_hash:
        raise HTTPException(status_code=400, detail="Interview has no audio")

    rich.print(interview)

    # Create a temporary file for the full audio
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        audio_data, _ = BLOBS.get(interview.audio_hash)
        tmp.write(audio_data)
        tmp.flush()
        tmp_path = Path(tmp.name)

        try:
            # Extract the next one-minute segment
            start_time = interview.current_position
            end_time = increment_time(start_time, seconds=60 * 2)

            # Extract audio for this segment
            segment_audio = await extract_segment(tmp_path, start_time, end_time)
            segment_hash = BLOBS.put(segment_audio, "audio/ogg")

            # Create new segment
            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                audio_hash=segment_hash,
                utterances=[],
            )

            # Upload segment to Gemini
            client = GeminiClient()
            file = await client.upload_bytes(
                segment_audio,
                mime_type="audio/ogg",
                display_name=f"segment_{start_time}",
            )

            rich.print(file)
            rich.print(f"bytes: {len(segment_audio)}")

            # Request transcription
            request = GenerateRequest(
                contents=[
                    Content(
                        role="user",
                        parts=[
                            Part(
                                fileData=FileData(
                                    fileUri=file.uri, mimeType="audio/ogg"
                                )
                            ),
                            Part(
                                text="""Please transcribe this audio segment.
                                Identify each speaker (S1, S2, etc.) and provide accurate timestamps.
                                Format your response as XML in the following format:
                                
                                <transcript>
                                  <utterance speaker="S1" start="00:00:03">Hello, how are you?</utterance>
                                  <utterance speaker="S2" start="00:00:05">I'm doing well, thank you.</utterance>
                                  ...
                                </transcript>

                                Make sure to:
                                1. Use speaker IDs like S1, S2, etc.
                                2. Include accurate timestamps in HH:MM:SS format
                                3. Put the spoken text as the content of each utterance tag
                                4. Only output valid XML - no other text before or after"""
                            ),
                        ],
                    )
                ],
            )

            response = await client.generate_content_sync(request)
            if not response.candidates:
                raise HTTPException(
                    status_code=500, detail="Failed to transcribe segment"
                )

            # Get the text content from the response
            full_text = response.candidates[0].content.parts[0].text

            # Find the XML content with a simple regexp
            xml_text = re.search(
                r"<transcript>(.*?)</transcript>", full_text, re.DOTALL
            )

            if not xml_text:
                raise HTTPException(
                    status_code=500, detail="Failed to find transcription XML"
                )

            assert isinstance(xml_text, re.Match)

            # Parse the XML into utterances
            new_utterances = parse_transcription_xml(xml_text.group(0))
            if not new_utterances:
                raise HTTPException(
                    status_code=500, detail="Failed to parse transcription XML"
                )

            # Add utterances to the segment
            segment.utterances = new_utterances

            # Update the current position based on the last utterance
            interview.current_position = increment_time(
                interview.current_position,
                seconds=60 * 2 - 1,
            )

            # Add the segment to the interview
            interview.segments.append(segment)

            # Save the updated interview
            INTERVIEWS.put(interview.id, interview)

        finally:
            # Clean up temp file
            tmp_path.unlink()
            # Clean up Gemini file
            await client.delete_file(file.name)

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


def increment_time(time_str: str, seconds: int) -> str:
    """Add seconds to a time string in HH:MM:SS format"""
    h, m, s = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s + seconds
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    import trio
    import hypercorn.trio
    import hypercorn.config
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    config = hypercorn.config.Config()
    config.bind = ["localhost:8000"]
    config.worker_class = "trio"

    trio.run(hypercorn.trio.serve, app, config)
