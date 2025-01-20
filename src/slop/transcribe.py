import hashlib
import logging
import re
import sqlite3
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import BinaryIO

import rich
import trio
from fastapi import FastAPI, UploadFile, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from slop.gemini import (
    Content,
    FileData,
    GenerateRequest,
    GeminiClient,
    Part,
)
from slop.views import upload_area
from tagflow import (
    DocumentMiddleware,
    Live,
    TagResponse,
    classes,
    tag,
    text,
    attr,
)

logger = logging.getLogger("slop.transcribe")

# Initialize Live instance for WebSocket support
live = Live()


class PartitionSegment(BaseModel):
    """A segment of the interview with start and end times."""

    start_time: str = Field(description="Start time of the segment (HH:MM:SS)")
    end_time: str = Field(description="End time of the segment (HH:MM:SS)")
    phrases: list[str] = Field(
        description="Some example short phrases from this segment to help orient the user"
    )
    audio_hash: str | None = None  # Hash of the segment's audio


class Utterance(BaseModel):
    """A single utterance in the interview."""

    speaker: str = Field(description="Speaker identifier (e.g. 'S1', 'S2')")
    timestamp: str = Field(description="Timestamp of the utterance (HH:MM:SS)")
    text: str = Field(description="The transcribed text")
    audio_hash: str | None = None  # Hash of the utterance's audio segment


class Segment(BaseModel):
    """A segment of the interview containing utterances."""

    start_time: str = Field(description="Start time of the segment (HH:MM:SS)")
    end_time: str = Field(description="End time of the segment (HH:MM:SS)")
    audio_hash: str | None = None  # Hash of the segment's audio
    utterances: list[Utterance] = []


class Interview(BaseModel):
    """An interview with its metadata and transcribed segments."""

    id: str
    filename: str
    file_uri: str
    audio_hash: str | None = None  # Hash of the processed audio file
    current_position: str = Field(
        default="00:00:00",
        description="Current position in the interview (HH:MM:SS)",
    )
    segments: list[Segment] = []


class Store:
    """
    A simple key-value store using SQLite and a Pydantic model.
    """

    def __init__(self, db_path: str | Path, model_class: type[BaseModel]):
        self.db_path = Path(db_path)
        self.model_class = model_class
        self._init_db()

    def _init_db(self):
        """Initialize the database with a key-value table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> BaseModel | None:
        """Get a value by key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store WHERE key = ?", (key,))
            if row := cursor.fetchone():
                return self.model_class.model_validate_json(row[0])
            return None

    def put(self, key: str, value: BaseModel):
        """Store a value by key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
                (key, value.model_dump_json()),
            )

    def values(self) -> list[BaseModel]:
        """Get all values."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store")
            return [
                self.model_class.model_validate_json(row[0])
                for row in cursor.fetchall()
            ]


class BlobStore:
    """
    Content-addressed store for binary data.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database with a blobs table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    mime_type TEXT NOT NULL
                )
                """
            )

    def put(self, data: bytes, mime_type: str) -> str:
        """Store binary data and return its hash."""
        hash_ = hashlib.sha256(data).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blobs (hash, data, mime_type) VALUES (?, ?, ?)",
                (hash_, data, mime_type),
            )
        return hash_

    def get(self, hash_: str) -> tuple[bytes, str] | None:
        """Get binary data and mime type by hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data, mime_type FROM blobs WHERE hash = ?", (hash_,)
            )
            if row := cursor.fetchone():
                return row[0], row[1]
            return None


# Initialize the stores
INTERVIEWS = Store(Path("data/interviews.db"), Interview)
BLOBS = BlobStore(Path("data/blobs.db"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager using the Live WebSocket support from TagFlow.
    """
    async with live.run(app):
        logger.info("Live server started")
        yield
    logger.info("Live server stopped")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=TagResponse,
    title="Slop Interview",
)
app.add_middleware(DocumentMiddleware)


@contextmanager
def layout(title: str):
    """Common layout wrapper for all pages."""
    with tag.html(lang="en"):
        with tag.head():
            with tag.title():
                text(f"{title} - Slop")

            # Tailwind + HTMX scripts
            with tag.script(src="https://cdn.tailwindcss.com"):
                pass
            with tag.script(src="https://unpkg.com/htmx.org@1.9.10"):
                pass
            with tag.script():
                text("""htmx.config = { "globalViewTransitions": true }""")
            with tag.script():
                text(
                    """
                    tailwind.config = {
                      theme: {
                        fontFamily: {
                            serif: ["Equity OT", "Times New Roman", "serif"],
                        },
                      },
                    }
                    """
                )
            live.script_tag()

        with tag.body(classes="bg-stone-300 min-h-screen font-serif p-4"):
            yield


@app.get("/")
async def home():
    """
    Renders the home page with an upload area.
    """
    with layout("Home"):
        with tag.div(classes="prose mx-auto"):
            upload_area()


async def process_audio(input_path: Path) -> bytes:
    """
    Convert audio to Ogg Vorbis format using ffmpeg.
    Returns the processed audio as bytes.
    """
    process = await trio.run_process(
        [
            "ffmpeg",
            "-i",
            str(input_path),  # Input file
            "-c:a",
            "libvorbis",  # Vorbis codec
            "-q:a",
            "6",  # Quality level (0-10, 4 is good)
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
    """
    Endpoint to handle file upload:
    1. Saves the file temporarily
    2. Processes and stores the audio in BlobStore
    3. Uploads to Gemini
    4. Creates an Interview record
    5. Redirects to the interview page
    """
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
            tmp_path.unlink()

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


@app.get("/interview/{interview_id}")
async def view_interview(interview_id: str):
    """
    Renders the interview page, showing the audio player and segments.
    """
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    with layout(f"Interview - {interview.filename}"):
        with tag.div(classes="prose mx-auto"):
            with tag.h1(classes="mb-2 text-bold"):
                # Display the filename minus extension
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

                            # Display each utterance
                            with tag.div(classes="flex flex-wrap gap-4 pl-4"):
                                for utterance in segment.utterances:
                                    with tag.span(
                                        **{"data-speaker": utterance.speaker}
                                    ):
                                        if utterance.speaker == "S1":
                                            classes("font-bold")
                                        text(utterance.text)

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


async def extract_segment(input_path: Path, start_time: str, end_time: str) -> bytes:
    """
    Extract a segment from an audio file using ffmpeg.
    Returns the extracted segment as bytes.
    """
    rich.print(
        {"input_path": str(input_path), "start_time": start_time, "end_time": end_time}
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


@app.get("/audio/{hash_}")
async def get_audio(hash_: str, request: Request):
    """
    Serve audio file by hash with support for range requests.
    """
    if not (result := BLOBS.get(hash_)):
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

    try:
        # Expected format: "bytes=start-end"
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

    content_length = end - start + 1

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


def parse_transcription_xml(xml_text: str) -> list[Utterance]:
    """
    Parse transcription XML into a list of Utterances.
    Expected format:
        <transcript>
            <utterance speaker="S1" start="00:00:03">Hello</utterance>
            ...
        </transcript>
    """
    import xml.etree.ElementTree as ET
    from io import StringIO

    try:
        tree = ET.parse(StringIO(xml_text))
        root = tree.getroot()
        utterances = []
        for utt in root.findall("utterance"):
            utterances.append(
                Utterance(
                    speaker=utt.get("speaker"),
                    timestamp=utt.get("start"),
                    text=utt.text.strip() if utt.text else "",
                )
            )
        return utterances
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        logger.error(f"XML content was: {xml_text}")
        return []


@app.post("/interview/{interview_id}/transcribe-next")
async def transcribe_next_segment(interview_id: str):
    """
    Transcribe the next audio segment (2 minutes) for the given interview:
    1. Extracts the next segment from the stored audio
    2. Uploads segment to Gemini for transcription
    3. Parses the transcription XML and appends it to the interview
    """
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    if not interview.audio_hash:
        raise HTTPException(status_code=400, detail="Interview has no audio")

    rich.print(interview)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        audio_data, _ = BLOBS.get(interview.audio_hash)
        tmp.write(audio_data)
        tmp.flush()
        tmp_path = Path(tmp.name)

        try:
            # Extract the next two-minute segment
            start_time = interview.current_position
            end_time = increment_time(start_time, seconds=60 * 2)

            segment_audio = await extract_segment(tmp_path, start_time, end_time)
            segment_hash = BLOBS.put(segment_audio, "audio/ogg")

            # Create a new segment
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

            # Prepare parts for the request
            parts = []

            # If there's a previous segment, include its audio and transcription first
            if interview.segments:
                prev_segment = interview.segments[-1]
                if prev_segment.audio_hash:
                    prev_audio_data, _ = BLOBS.get(prev_segment.audio_hash)
                    prev_file = await client.upload_bytes(
                        prev_audio_data,
                        mime_type="audio/ogg",
                        display_name=f"previous_segment_{prev_segment.start_time}",
                    )
                    parts.extend(
                        [
                            Part(
                                text="For context, here is the previous segment's audio and transcription:"
                            ),
                            Part(
                                fileData=FileData(
                                    fileUri=prev_file.uri, mimeType="audio/ogg"
                                )
                            ),
                            Part(
                                text="".join(
                                    f'<utterance speaker="{u.speaker}" start="{u.timestamp}">{u.text}</utterance>'
                                    for u in prev_segment.utterances
                                )
                            ),
                        ]
                    )

            # Add current segment's audio and instructions
            parts.extend(
                [
                    Part(text="Please transcribe this audio segment:"),
                    Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                    Part(
                        text="""Format your response as XML in the following format:

<transcript>
  <utterance speaker="S1" start="00:00:03">Hello, how are you?</utterance>
  <utterance speaker="S2" start="00:00:05">I'm doing well, thank you.</utterance>
</transcript>

1. Use speaker IDs like S1, S2, etc.
2. Include timestamps in HH:MM:SS format
3. Output only valid XML, no extra text.
4. Maintain consistent speaker identities with the previous segment's context.
"""
                    ),
                ]
            )

            # Request transcription
            request = GenerateRequest(
                contents=[
                    Content(
                        role="user",
                        parts=parts,
                    )
                ]
            )

            response = await client.generate_content_sync(request)
            if not response.candidates:
                raise HTTPException(
                    status_code=500, detail="Failed to transcribe segment"
                )

            # Extract the text content
            full_text = response.candidates[0].content.parts[0].text

            # Find the XML
            xml_match = re.search(
                r"<transcript>(.*?)</transcript>", full_text, re.DOTALL
            )
            if not xml_match:
                raise HTTPException(
                    status_code=500, detail="Failed to find transcription XML"
                )

            xml_text = xml_match.group(0)
            new_utterances = parse_transcription_xml(xml_text)
            if not new_utterances:
                raise HTTPException(
                    status_code=500, detail="Failed to parse transcription XML"
                )

            # Add utterances to the segment
            segment.utterances = new_utterances

            # Advance current position by two minutes - 1 second
            interview.current_position = increment_time(start_time, seconds=60 * 2 - 1)

            # Append segment to interview and save
            interview.segments.append(segment)
            INTERVIEWS.put(interview.id, interview)
        finally:
            tmp_path.unlink()
            await client.delete_file(file.name)

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


def increment_time(time_str: str, seconds: int) -> str:
    """Add `seconds` to a time string in HH:MM:SS format."""
    h, m, s_ = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s_ + seconds
    h, remainder = divmod(total_seconds, 3600)
    m, s_ = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s_:02d}"


def main():
    """
    Runs the application with Hypercorn and Trio.
    """
    import hypercorn.config
    import hypercorn.trio

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


if __name__ == "__main__":
    main()
