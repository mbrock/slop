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
from fastapi import FastAPI, UploadFile, HTTPException, Request, Response, Form
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
    duration: str = Field(
        default="00:00:00",
        description="Total duration of the interview (HH:MM:SS)",
    )
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
            with tag.script(src="https://unpkg.com/htmx.org@2.0.4"):
                pass
            live.script_tag()

            # Add CSS for loading indicator
            with tag.style():
                text("""
                    .htmx-indicator {
                        opacity: 0;
                        transition: opacity 200ms ease-in;
                    }
                    .htmx-request .htmx-indicator {
                        opacity: 1
                    }
                    .htmx-request.htmx-indicator {
                        opacity: 1
                    }
                """)

            with tag.script(
                type="module", src="https://cdn.jsdelivr.net/npm/media-chrome@3/+esm"
            ):
                pass

        with tag.body(classes="min-h-screen"):
            svg_icons()
            yield


def time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to total seconds."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def calculate_progress(current: str, total: str) -> int:
    """Calculate progress percentage."""
    current_secs = time_to_seconds(current)
    total_secs = time_to_seconds(total)
    if total_secs == 0:
        return 0
    return int((current_secs / total_secs) * 100)


@app.get("/")
async def home():
    """
    Renders the home page with an upload area.
    """
    with layout("Home"):
        with tag.div(classes="max-w-4xl mx-auto p-4"):
            with tag.div(classes="mb-8"):
                with tag.h1(classes="text-2xl font-bold mb-4"):
                    text("Interviews")

                with tag.div(classes="space-y-4"):
                    for interview in INTERVIEWS.values():
                        progress = calculate_progress(
                            interview.current_position, interview.duration
                        )
                        with tag.div(classes="border rounded-lg p-4 hover:bg-gray-50"):
                            with tag.a(
                                href=f"/interview/{interview.id}",
                                classes="block",
                            ):
                                with tag.div(
                                    classes="flex justify-between items-center mb-2"
                                ):
                                    with tag.span(classes="font-medium"):
                                        text(interview.filename)
                                    with tag.span(classes="text-gray-500 text-sm"):
                                        text(
                                            f"{interview.current_position} / {interview.duration}"
                                        )

                                with tag.div(classes="bg-gray-200 rounded-full h-2"):
                                    with tag.div(
                                        classes="bg-blue-600 rounded-full h-2 transition-all",
                                        style=f"width: {progress}%",
                                    ):
                                        pass

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
            "8",  # Quality level (0-10, 4 is good)
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
            # Get audio duration
            duration = await get_audio_duration(tmp_path)

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
                duration=duration,
            )
            INTERVIEWS.put(interview_id, interview)
        finally:
            tmp_path.unlink()

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


def svg_icons():
    with tag("svg", classes="hidden"):
        # backward symbol
        with tag(
            "symbol",
            id="backward",
            viewBox="0 0 24 24",
            **{
                "stroke-width": "1.5",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        ):
            with tag(
                "path",
                d="M8 5L5 8M5 8L8 11M5 8H13.5C16.5376 8 19 10.4624 19 13.5C19 15.4826 18.148 17.2202 17 18.188",
            ):
                pass
            with tag("path", d="M5 15V19"):
                pass
            with tag(
                "path",
                d="M8 18V16C8 15.4477 8.44772 15 9 15H10C10.5523 15 11 15.4477 11 16V18C11 18.5523 10.5523 19 10 19H9C8.44772 19 8 18.5523 8 18Z",
            ):
                pass

        # play symbol
        with tag("symbol", id="play", viewBox="0 0 24 24"):
            with tag(
                "path",
                **{
                    "fill-rule": "evenodd",
                    "clip-rule": "evenodd",
                    "d": "M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0\
        3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z",
                },
            ):
                pass

        # pause symbol
        with tag("symbol", id="pause", viewBox="0 0 24 24"):
            with tag(
                "path",
                **{
                    "fill-rule": "evenodd",
                    "clip-rule": "evenodd",
                    "d": "M6.75 5.25a.75.75 0 01.75-.75H9a.75.75 0 01.75.75v13.5a.75.75 0\
      01-.75.75H7.5a.75.75 0 01-.75-.75V5.25zm7.5 0A.75.75 0 0115 4.5h1.5a.75.75 0 01.75.75v13.5a.75.75 0\
      01-.75.75H15a.75.75 0 01-.75-.75V5.25z",
                },
            ):
                pass

        # forward symbol
        with tag(
            "symbol",
            id="forward",
            viewBox="0 0 24 24",
            **{
                "stroke-width": "1.5",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        ):
            with tag(
                "path",
                d="M16 5L19 8M19 8L16 11M19 8H10.5C7.46243 8 5 10.4624 5 13.5C5 15.4826 5.85204 17.2202 7 18.188",
            ):
                pass
            with tag("path", d="M13 15V19"):
                pass
            with tag(
                "path",
                d="M16 18V16C16 15.4477 16.4477 15 17 15H18C18.5523 15 19 15.4477 19 16V18C19 18.5523 18.5523 19 18\
      19H17C16.4477 19 16 18.5523 16 18Z",
            ):
                pass

        # high symbol
        with tag("symbol", id="high", viewBox="0 0 24 24"):
            with tag(
                "path",
                d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0\
      001.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276\
      2.561-1.06V4.06zM18.584 5.106a.75.75 0 011.06 0c3.808 3.807 3.808 9.98 0 13.788a.75.75 0 11-1.06-1.06\
      8.25 8.25 0 000-11.668.75.75 0 010-1.06z",
            ):
                pass
            with tag(
                "path",
                d="M15.932 7.757a.75.75 0 011.061 0 6 6 0 010 8.486.75.75 0 01-1.06-1.061 4.5 4.5 0 000-6.364.75.75 0\
      010-1.06z",
            ):
                pass

        # off symbol
        with tag("symbol", id="off", viewBox="0 0 24 24"):
            with tag(
                "path",
                d="M13.5 4.06c0-1.336-1.616-2.005-2.56-1.06l-4.5 4.5H4.508c-1.141 0-2.318.664-2.66 1.905A9.76 9.76 0\
      001.5 12c0 .898.121 1.768.35 2.595.341 1.24 1.518 1.905 2.659 1.905h1.93l4.5 4.5c.945.945 2.561.276\
      2.561-1.06V4.06zM17.78 9.22a.75.75 0 10-1.06 1.06L18.44 12l-1.72 1.72a.75.75 0 001.06 1.06l1.72-1.72 1.72\
      1.72a.75.75 0 101.06-1.06L20.56 12l1.72-1.72a.75.75 0 00-1.06-1.06l-1.72 1.72-1.72-1.72z",
            ):
                pass


def audio_player(src: str):
    with tag("media-controller", audio=True, classes="w-full"):
        attr(
            "style",
            """
            --media-background-color: transparent;
            --media-control-background: transparent;
            --media-control-hover-background: transparent;
        """,
        )
        with tag("audio", slot="media", src=src, crossorigin=True):
            pass
        with tag(
            "media-control-bar",
            classes="h-12 w-full bg-white items-center",
        ):
            # rounded-md ring-1 ring-slate-700/10 shadow-xl shadow-black/5
            with tag("media-seek-backward-button", classes="p-0"):
                with tag.svg(
                    slot="icon",
                    aria_hidden=True,
                    classes="w-7 h-7 fill-none stroke-gray-500",
                ):
                    with tag.use(href="#backward"):
                        pass
            with tag(
                "media-play-button",
                classes="h-7 w-7 p-2 mx-3 rounded-full bg-gray-700",
            ):
                with tag.svg(slot="play", aria_hidden=True, classes="relative left-px"):
                    with tag.use(href="#play"):
                        pass
                with tag.svg(slot="pause", aria_hidden=True):
                    with tag.use(href="#pause"):
                        pass
            with tag("media-seek-forward-button", classes="p-0"):
                with tag.svg(
                    slot="icon",
                    aria_hidden=True,
                    classes="w-7 h-7 fill-none stroke-gray-500",
                ):
                    with tag.use(href="#forward"):
                        pass
            with tag("media-time-display", classes="text-gray-500 text-sm"):
                pass
            with tag(
                "media-time-range",
                classes="block h-2 min-h-0 p-0 m-2 rounded-md bg-gray-50",
            ):
                attr(
                    "style",
                    """
                    --media-range-track-background: transparent;
                    --media-time-range-buffered-color: rgb(0 0 0 / 0.02);
                    --media-range-bar-color: rgb(79 70 229);
                    --media-range-track-border-radius: 4px;
                    --media-range-track-height: 0.5rem;
                    --media-range-thumb-background: rgb(79 70 229);
                    --media-range-thumb-box-shadow: 0 0 0 2px rgb(255 255 255 / 0.9);
                    --media-range-thumb-width: 0.25rem;
                    --media-range-thumb-height: 1rem;
                    --media-preview-time-text-shadow: transparent;
                """,
                )
                with tag("media-preview-time-display", classes="text-gray-600 text-xs"):
                    pass
            with tag("media-duration-display", classes="text-gray-500 text-xs"):
                pass
            with tag("media-mute-button"):
                with tag.svg(
                    slot="high",
                    aria_hidden=True,
                    classes="w-5 h-5 fill-gray-500",
                ):
                    with tag.use(href="#high"):
                        pass
                with tag.svg(
                    slot="medium",
                    aria_hidden=True,
                    classes="w-5 h-5 fill-gray-500",
                ):
                    with tag.use(href="#high"):
                        pass
                with tag.svg(
                    slot="low",
                    aria_hidden=True,
                    classes="w-5 h-5 fill-gray-500",
                ):
                    with tag.use(href="#high"):
                        pass
                with tag.svg(
                    slot="off",
                    aria_hidden=True,
                    classes="w-5 h-5 fill-gray-500",
                ):
                    with tag.use(href="#off"):
                        pass


def breadcrumb(items: dict[str, str]):
    with tag.nav(classes="hidden sm:flex", aria_label="Breadcrumb"):
        with tag.ol(role="list", classes="flex items-center space-x-4"):
            total = len(items)
            for i, (label, href) in enumerate(items.items()):
                with tag.li():
                    with tag.div(classes="flex items-center"):
                        # Render the arrow separator only if it's not the first item
                        if i > 0:
                            with tag.svg(
                                classes="size-5 shrink-0 text-gray-400 mr-4",
                                viewBox="0 0 20 20",
                                fill="currentColor",
                                aria_hidden="true",
                                **{"data-slot": "icon"},
                            ):
                                with tag.path(
                                    **{
                                        "fill-rule": "evenodd",
                                        "d": "M8.22 5.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L11.94 10 8.22 6.28a.75.75 0 0 1 0-1.06Z",
                                        "clip-rule": "evenodd",
                                    }
                                ):
                                    pass
                        with tag.a(
                            href=href,
                            classes="text-sm font-medium text-gray-500 hover:text-gray-700",
                            aria_current="page" if i == (total - 1) else None,
                        ):
                            text(label)


def interview_header(title: str, interview_id: str):
    """Renders the interview header with title and action buttons."""
    with tag.div(classes="px-4 py-2"):
        breadcrumb({"Interviews": "/", title: "#"})
        with tag.div(classes="mt-2 md:flex md:items-center md:justify-between"):
            with tag.div(classes="min-w-0 flex-1"):
                with tag.h2(
                    classes="text-2xl/7 font-bold text-gray-900 sm:truncate sm:text-3xl sm:tracking-tight"
                ):
                    text(title)
            with tag.div(classes="mt-4 flex shrink-0 md:ml-4 md:mt-0"):
                with tag.form(
                    action=f"/interview/{interview_id}/rename",
                    method="post",
                    onsubmit="const name = prompt('New name:', this.querySelector('input').value); if (!name) return false; this.querySelector('input').value = name;",
                ):
                    with tag.input(type="hidden", name="new_name", value=title):
                        pass
                    with tag.button(
                        type="submit",
                        classes="inline-flex items-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50",
                    ):
                        text("Rename")


@app.get("/interview/{interview_id}/segment/{segment_index}")
async def view_segment(interview_id: str, segment_index: int):
    """Renders a single segment as a partial view."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")

    with tag.div(
        id=f"segment-{segment_index}",
        classes="flex flex-col gap-2 p-4 py-2 border-t-4 border-gray-400",
    ):
        with tag.div(classes="flex items-center gap-2"):
            if segment.audio_hash:
                audio_player(f"/audio/{segment.audio_hash}")

            # Add edit button
            with tag.button(
                classes="px-2 py-1 text-sm text-gray-600 hover:text-gray-900",
                **{
                    "hx-get": f"/interview/{interview_id}/segment/{segment_index}/edit",
                    "hx-target": f"#segment-content-{segment_index}",
                    "hx-swap": "innerHTML",
                },
            ):
                text("Edit")

        # Display each utterance
        with tag.div(id=f"segment-content-{segment_index}", classes="w-full"):
            with tag.div(classes="flex flex-wrap gap-4"):
                for utterance in segment.utterances:
                    with tag.span(**{"data-speaker": utterance.speaker}):
                        if utterance.speaker == "S1":
                            classes("font-bold")
                        text(utterance.text)


@app.get("/interview/{interview_id}/segment/{segment_index}/edit")
async def edit_segment_dialog(interview_id: str, segment_index: int):
    """Renders the inline edit form for a segment."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Convert utterances to text format
    text_content = "\n\n".join(f"{u.speaker}: {u.text}" for u in segment.utterances)

    with tag.form(
        classes="space-y-4",
        **{
            "hx-put": f"/interview/{interview_id}/segment/{segment_index}",
            "hx-target": f"#segment-content-{segment_index}",
            "hx-swap": "innerHTML",
        },
    ):
        with tag.div(classes="flex gap-2"):
            with tag.textarea(
                name="content",
                classes="flex-1 h-32 font-mono p-2 border rounded text-sm",
                placeholder="Format: 'S1: Hello\n\nS2: Hi there'",
            ):
                text(text_content)
            with tag.div(classes="flex flex-col gap-2"):
                with tag.button(
                    type="submit",
                    classes="p-2 text-blue-600 hover:text-blue-800",
                ):
                    with tag.svg(
                        xmlns="http://www.w3.org/2000/svg",
                        fill="none",
                        viewBox="0 0 24 24",
                        **{"stroke-width": "1.5"},
                        stroke="currentColor",
                        classes="w-5 h-5",
                    ):
                        with tag.path(
                            **{
                                "stroke-linecap": "round",
                                "stroke-linejoin": "round",
                                "d": "m4.5 12.75 6 6 9-13.5",
                            }
                        ):
                            pass
                with tag.button(
                    type="button",
                    classes="p-2 text-gray-600 hover:text-gray-800",
                    **{
                        "hx-get": f"/interview/{interview_id}/segment/{segment_index}",
                        "hx-target": f"#segment-{segment_index}",
                        "hx-swap": "outerHTML",
                    },
                ):
                    with tag.svg(
                        xmlns="http://www.w3.org/2000/svg",
                        fill="none",
                        viewBox="0 0 24 24",
                        **{"stroke-width": "1.5"},
                        stroke="currentColor",
                        classes="w-5 h-5",
                    ):
                        with tag.path(
                            **{
                                "stroke-linecap": "round",
                                "stroke-linejoin": "round",
                                "d": "M6 18 18 6M6 6l12 12",
                            }
                        ):
                            pass


@app.get("/interview/{interview_id}")
async def view_interview(interview_id: str):
    """
    Renders the interview page, showing the audio player and segments.
    """
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    with layout(f"Interview - {interview.filename}"):
        with tag.div(classes="prose mx-auto"):
            interview_header(interview.filename, interview.id)

            # if interview.audio_hash:
            #     with tag.div(classes="mb-2"):
            #         with tag.audio(
            #             src=f"/audio/{interview.audio_hash}",
            #             controls=True,
            #             classes="w-full",
            #             preload="metadata",
            #         ):
            #             pass

            with tag.div(id="segments", classes="flex flex-col gap-2"):
                if interview.segments:
                    for i, segment in enumerate(interview.segments):
                        await view_segment(interview_id, i)

            # Progress bar and transcribe button section
            with tag.div(classes="border-t-4 border-gray-400 mt-4"):
                with tag.div(classes="flex items-center justify-between p-4"):
                    # Progress info
                    with tag.div(classes="flex items-center gap-2"):
                        with tag.span(classes="text-gray-500 text-sm"):
                            text(f"{interview.current_position} / {interview.duration}")
                        with tag.div(classes="w-48 bg-gray-200 rounded-full h-2"):
                            progress = calculate_progress(
                                interview.current_position, interview.duration
                            )
                            with tag.div(
                                classes="bg-blue-600 rounded-full h-2 transition-all",
                                style=f"width: {progress}%",
                            ):
                                pass

                    # Transcribe button
                    with tag.button(
                        classes="flex items-center gap-2 text-blue-600 hover:text-blue-800 font-medium",
                        **{
                            "hx-post": f"/interview/{interview_id}/transcribe-next",
                            "hx-target": "#segments",
                            "hx-swap": "beforeend",
                        },
                    ):
                        text("Transcribe next segment")
                        with tag.span(classes="htmx-indicator"):
                            text("🤔")


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
            "8",  # Quality level
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
            <utterance speaker="S1">Hello</utterance>
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
    Transcribe the next audio segment (2 minutes) for the given interview.
    Returns just the new segment as a partial view.
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
                            Part(text="Previous segment's audio and transcription:"),
                            Part(
                                fileData=FileData(
                                    fileUri=prev_file.uri, mimeType="audio/ogg"
                                )
                            ),
                            Part(
                                text="".join(
                                    f'<utterance speaker="{u.speaker}">{u.text}</utterance>'
                                    for u in prev_segment.utterances
                                )
                            ),
                        ]
                    )

            # Add current segment's audio and instructions
            parts = [
                *parts,
                Part(text="New audio segment:"),
                Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                Part(
                    text="""Format your response as XML in the following format:

<transcript>
  <utterance speaker="S1">Hello, how are you?</utterance>
  <utterance speaker="S2">I'm doing— I'm— yeah, I'm great.</utterance>
</transcript>

1. Use speaker IDs like S1, S2, etc.
2. Output only valid XML, no extra text.
3. Maintain consistent speaker identities with the previous segment's context.
4. Use em dashes (—) for interruptions and disfluencies.
"""
                ),
            ]

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

            # Return the new segment as a partial view
            return await view_segment(interview_id, len(interview.segments) - 1)
        finally:
            tmp_path.unlink()
            await client.delete_file(file.name)


def increment_time(time_str: str, seconds: int) -> str:
    """Add `seconds` to a time string in HH:MM:SS format."""
    h, m, s_ = map(int, time_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s_ + seconds
    h, remainder = divmod(total_seconds, 3600)
    m, s_ = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s_:02d}"


async def get_audio_duration(input_path: Path) -> str:
    """
    Get the duration of an audio file using ffprobe.
    Returns duration in HH:MM:SS format.
    """
    process = await trio.run_process(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_stdout=True,
        capture_stderr=True,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFprobe failed: {stderr}")

    duration_secs = float(process.stdout.decode().strip())
    hours = int(duration_secs // 3600)
    minutes = int((duration_secs % 3600) // 60)
    seconds = int(duration_secs % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@app.put("/interview/{interview_id}/segment/{segment_index}")
async def update_segment(
    interview_id: str,
    segment_index: int,
    content: str = Form(...),
):
    """Updates a segment's utterances from the edit form."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Parse the content into utterances
    utterances = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        speaker, content = line.split(":", 1)
        utterances.append(
            Utterance(
                speaker=speaker.strip(),
                text=content.strip(),
            )
        )

    # Update the segment
    segment.utterances = utterances
    INTERVIEWS.put(interview_id, interview)

    # Return the updated segment view
    with tag.div(classes="flex flex-wrap gap-4"):
        for utterance in segment.utterances:
            with tag.span(**{"data-speaker": utterance.speaker}):
                if utterance.speaker == "S1":
                    classes("font-bold")
                text(utterance.text)


@app.post("/interview/{interview_id}/rename")
async def rename_interview(interview_id: str, new_name: str = Form(...)):
    """Processes the interview rename."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    interview.filename = new_name
    INTERVIEWS.put(interview_id, interview)

    return RedirectResponse(url=f"/interview/{interview_id}", status_code=303)


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
