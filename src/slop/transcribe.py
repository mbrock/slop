import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager, contextmanager
from io import BytesIO
from pathlib import Path
from subprocess import PIPE

import anyio
import httpx
from docx import Document
from docx.shared import Pt
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from tagflow import (
    DocumentMiddleware,
    Live,
    TagResponse,
    attr,
    classes,
    tag,
    text,
)

from slop import gemini
from slop.gemini import GeminiError, ModelOverloadedError
from slop.models import (
    BLOBS,
    INTERVIEWS,
    Interview,
    Segment,
    Utterance,
)
from slop.sndprompt import (
    improve_speaker_identification_segment,
    transcribe_audio_segment,
)
from slop.views import speaker_classes, upload_area

logger = logging.getLogger("slop.transcribe")

# Initialize Live instance for WebSocket support
live = Live()

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]


IMPROVE_SPEAKERS_ONCLICK = (
    "const hint = prompt('Enter any hints about the speakers (optional):'); "
    "if (hint === null) { return false; } "
    "const selected = document.querySelector(\"#model-selector input[name='model_name']:checked\"); "
    "const payload = { hint }; "
    "if (selected) { payload.model_name = selected.value; } "
    "this.setAttribute('hx-vals', JSON.stringify(payload)); "
    "return true;"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager using the Live WebSocket support from TagFlow.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable must be set")

    async with httpx.AsyncClient() as client:
        with gemini.http_client.using(client):
            with gemini.api_key.using(api_key):
                with gemini.model.using(DEFAULT_GEMINI_MODEL):
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
            with tag.script(src="https://unpkg.com/hyperscript.org@0.9.12"):
                pass
            live.script_tag()

            # Add key tracking script
            with tag.script():
                text("""
                    window.keyPressed = undefined;
                    document.addEventListener('keydown', (e) => {
                        window.keyPressed = e.key;
                        console.log(window.keyPressed, "pressed");
                    });
                    document.addEventListener('keyup', () => {
                        window.keyPressed = undefined;
                        console.log("key released");
                    });
                """)

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

                    button.htmx-request {
                        opacity: 0.5;
                        cursor: wait;
                    }
                """)

            with tag.script(
                type="module", src="https://cdn.jsdelivr.net/npm/media-chrome@3/+esm"
            ):
                pass

        with tag.body(classes="min-h-screen"):
            svg_icons()
            with tag.main():
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


@app.get("/interview-list")
def interview_list():
    """
    Renders the interview list page.
    """
    with tag.div(classes="space-y-4", id="interview-list"):
        for interview in INTERVIEWS.values():
            progress = calculate_progress(
                interview.current_position, interview.duration
            )
            with tag.div(classes="border rounded-lg p-4 hover:bg-gray-50"):
                with tag.a(
                    href=f"/interview/{interview.id}",
                    classes="block",
                ):
                    with tag.div(classes="flex justify-between items-center mb-2"):
                        with tag.span(classes="font-medium"):
                            text(interview.filename)
                        with tag.span(classes="text-gray-500 text-sm"):
                            text(f"{interview.current_position} / {interview.duration}")

                    with tag.div(classes="bg-gray-200 rounded-full h-2"):
                        with tag.div(
                            classes="bg-blue-600 rounded-full h-2 transition-all",
                            style=f"width: {progress}%",
                        ):
                            pass


@app.get("/home")
def render_home():
    breadcrumb({"Ieva's Interviews": "#"})
    with tag.div(classes="max-w-4xl mx-auto p-4"):
        with tag.div(classes="mb-8"):
            # with tag.h1(classes="text-2xl font-bold mb-4"):
            #     text("Ieva's Interviews")

            interview_list()

        with tag.div(classes="prose mx-auto"):
            upload_area(target="main")


@app.get("/")
async def home():
    """
    Renders the home page with an upload area.
    """
    with layout("Home"):
        render_home()


async def process_audio(input_path: Path) -> bytes:
    """
    Convert audio to Ogg Vorbis format using ffmpeg.
    Returns the processed audio as bytes.
    """
    logger.info(f"Processing audio file: {input_path}")
    process = await anyio.run_process(
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
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    logger.info(f"FFmpeg process {process.returncode}")
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
    3. Creates an Interview record
    4. Redirects to the interview page
    """
    upload_name = audio.filename or "upload.ogg"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(upload_name).suffix
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

            # Create interview record
            interview_id = str(len(list(INTERVIEWS.values())) + 1)
            interview = Interview(
                id=interview_id,
                filename=upload_name,
                audio_hash=audio_hash,
                duration=duration,
            )
            INTERVIEWS.put(interview_id, interview)
        finally:
            tmp_path.unlink()

    render_home()


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
    with tag.nav(
        classes="hidden sm:flex bg-gray-300 px-4 py-1 border-b border-gray-400",
        aria_label="Breadcrumb",
    ):
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
                            classes="text-sm font-medium text-gray-600 hover:text-gray-700",
                            aria_current="page" if i == (total - 1) else None,
                        ):
                            text(label)


def button_view(label: str, href: str | None = None, type: str = "button", **attrs):
    """Renders a button with consistent styling."""
    base_classes = "inline-flex items-center rounded-md px-3 py-1 text-sm font-semibold text-gray-900 bg-white shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 whitespace-nowrap"

    if href:
        with tag.a(href=href, classes=base_classes):
            text(label)
    else:
        with tag.button(type=type, classes=base_classes, **attrs):
            text(label)


def interview_header(title: str, interview_id: str):
    """Renders the interview header with title and action buttons."""
    interview = INTERVIEWS.get(interview_id)
    context_segments_value = str(interview.context_segments) if interview else "0"

    with tag.div():
        breadcrumb({"Ieva's Interviews": "/", title: "#"})
        with tag.div(
            classes="mt-2 md:flex md:items-center md:justify-between px-4 py-2"
        ):
            with tag.div(classes="min-w-0 flex-1"):
                with tag.h2(
                    classes="text-2xl/7 font-bold text-gray-900 sm:truncate sm:text-3xl sm:tracking-tight"
                ):
                    text(title)
            with tag.div(classes="mt-4 flex shrink-0 gap-4 md:ml-4 md:mt-0"):
                # Context segments control
                with tag.form(
                    action=f"/interview/{interview_id}/context-segments",
                    method="post",
                    classes="flex items-center gap-2",
                ):
                    with tag.label(classes="text-sm text-gray-600"):
                        text("Context segments:")
                    with tag.input(
                        type="number",
                        name="context_segments",
                        min="0",
                        max="5",
                        value=context_segments_value,
                        classes="w-16 px-2 py-1 text-sm border rounded",
                        **{
                            "hx-post": f"/interview/{interview_id}/context-segments",
                            "hx-trigger": "change",
                            "hx-swap": "none",
                        },
                    ):
                        pass

                # Model selector
                with tag.form(
                    id="model-selector",
                    classes="flex items-center gap-2",
                ):
                    with tag.label(classes="text-sm text-gray-600"):
                        text("Model:")
                    with tag.div(classes="flex gap-2"):
                        for model in MODEL_CHOICES:
                            with tag.label(classes="flex items-center gap-1"):
                                with tag.input(
                                    type="radio",
                                    name="model_name",
                                    value=model,
                                    checked=(model == gemini.model.get()),
                                    classes="text-blue-600",
                                ):
                                    pass
                                with tag.span(classes="text-sm text-gray-600"):
                                    text(model)

                with tag.form(
                    action=f"/interview/{interview_id}/rename",
                    method="post",
                    onsubmit="const name = prompt('New name:', this.querySelector('input').value); if (!name) return false; this.querySelector('input').value = name;",
                ):
                    with tag.input(type="hidden", name="new_name", value=title):
                        pass
                    button_view("Rename", type="submit")
                button_view("Export DOCX", href=f"/interview/{interview_id}/export")


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
        classes="flex flex-col gap-2 p-4 py-2 border-t-4 border-gray-400 mt-4",
    ):
        with tag.div(classes="flex items-center gap-2"):
            if segment.audio_hash:
                audio_player(f"/audio/{segment.audio_hash}")

            # Add edit button
            button_view(
                "Edit",
                **{
                    "hx-get": f"/interview/{interview_id}/segment/{segment_index}/edit",
                    "hx-target": f"#segment-content-{segment_index}",
                    "hx-swap": "innerHTML",
                },
            )

            # Add retranscribe button
            button_view(
                "Retranscribe",
                **{
                    "hx-post": f"/interview/{interview_id}/segment/{segment_index}/retranscribe",
                    "hx-target": f"#segment-{segment_index}",
                    "hx-swap": "outerHTML",
                    "hx-include": "#model-selector",
                },
            )

            # Add improve speakers button
            button_view(
                "Improve Speakers",
                **{
                    "hx-post": f"/interview/{interview_id}/segment/{segment_index}/improve-speakers",
                    "hx-target": f"#segment-{segment_index}",
                    "hx-swap": "outerHTML",
                    "onclick": IMPROVE_SPEAKERS_ONCLICK,
                },
            )

        # Display each utterance
        with tag.div(id=f"segment-content-{segment_index}", classes="w-full"):
            with tag.div(classes="flex flex-wrap gap-4"):
                for i, utterance in enumerate(segment.utterances):
                    render_utterance(interview_id, segment_index, i, utterance)


def render_utterance(interview_id, segment_index, i, utterance):
    with tag.span(
        **{
            "data-speaker": utterance.speaker,
            "hx-post": f"/interview/{interview_id}/segment/{segment_index}/update-speaker",
            "hx-trigger": "click",
            "hx-vals": f'js:{{"utterance_index": {i}, "key": window.keyPressed}}',
            "hx-swap": "outerHTML",
        }
    ):
        classes(speaker_classes(utterance.speaker), "hover:underline")
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
                classes="flex-1 h-[60vh] font-mono p-2 border rounded text-sm",
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
                    button_view(
                        "Transcribe more",
                        **{
                            "hx-post": f"/interview/{interview_id}/transcribe-next",
                            "hx-target": "#segments",
                            "hx-swap": "beforeend",
                            "hx-include": "#model-selector",
                        },
                    )


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


@app.post("/interview/{interview_id}/transcribe-next")
async def transcribe_next_segment(
    interview_id: str,
):
    """
    Transcribe the next audio segment (2 minutes) for the given interview.
    Returns just the new segment as a partial view.
    """
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    # Get context segments
    context_segments = []
    if interview.segments:
        start_idx = max(0, len(interview.segments) - interview.context_segments)
        context_segments = interview.segments[start_idx:]

    # Calculate segment times
    start_time = interview.current_position
    end_time = increment_time(start_time, seconds=60 * 2)

    try:
        # Transcribe the segment
        utterances, segment_hash = await transcribe_audio_segment(
            interview_id,
            start_time,
            end_time,
            context_segments,
        )

        # Create and save the new segment
        segment = Segment(
            start_time=start_time,
            end_time=end_time,
            audio_hash=segment_hash,
            utterances=utterances,
        )

        # Advance current position by two minutes - 1 second
        interview.current_position = increment_time(start_time, seconds=60 * 2 - 1)

        # Append segment to interview and save
        interview.segments.append(segment)
        INTERVIEWS.put(interview_id, interview)

        # Return the new segment as a partial view
        return await view_segment(interview_id, len(interview.segments) - 1)

    except ModelOverloadedError as e:
        with tag.div(
            classes="fixed bottom-4 right-4 bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded shadow-lg",
            role="alert",
            **{
                "hx-swap-oob": "true",
                "_": "on load wait 5s then remove me",
            },
        ):
            with tag.p(classes="font-bold"):
                text("Model overloaded")
            with tag.p():
                text(
                    f"{gemini.model.get()} is busy. Try switching to {e.alternative_model} and retry."
                )

    except GeminiError as e:
        # Show error toast notification
        with tag.div(
            classes="fixed bottom-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded shadow-lg",
            role="alert",
            **{
                "hx-swap-oob": "true",
                "_": "on load wait 5s then remove me",
            },
        ):
            with tag.p(classes="font-bold"):
                text("Error")
            with tag.p():
                text(e.message)


@app.post("/interview/{interview_id}/segment/{segment_index}/retranscribe")
async def retranscribe_segment(
    interview_id: str,
    segment_index: int,
):
    """Retranscribe a specific segment using Gemini."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Get context segments
    context_segments = []
    if segment_index > 0:
        start_idx = max(0, segment_index - interview.context_segments)
        context_segments = interview.segments[start_idx:segment_index]

    # Transcribe the segment
    utterances, segment_hash = await transcribe_audio_segment(
        interview_id,
        segment.start_time,
        segment.end_time,
        context_segments,
    )

    # Update the segment
    segment.audio_hash = segment_hash
    segment.utterances = utterances
    INTERVIEWS.put(interview_id, interview)

    # Return the updated segment view
    return await view_segment(interview_id, segment_index)


@app.post("/interview/{interview_id}/segment/{segment_index}/improve-speakers")
async def improve_speaker_identification(
    interview_id: str,
    segment_index: int,
    hint: str | None = Form(None),
):
    """Improve speaker identification for a specific segment using Gemini."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Get context segments
    context_segments = []
    if segment_index > 0:
        start_idx = max(0, segment_index - interview.context_segments)
        context_segments = interview.segments[start_idx:segment_index]

    # Get the segment audio
    if not segment.audio_hash:
        raise HTTPException(status_code=400, detail="Segment has no audio")

    segment_blob = BLOBS.get(segment.audio_hash)
    if not segment_blob:
        raise HTTPException(status_code=404, detail="Segment audio not found")

    audio_data, _ = segment_blob

    # Improve speaker identification
    utterances = await improve_speaker_identification_segment(
        audio_data,
        segment.utterances,
        context_segments,
        hint,
    )

    # Update the segment
    segment.utterances = utterances
    INTERVIEWS.put(interview_id, interview)

    # Return the updated segment view
    return await view_segment(interview_id, segment_index)


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
    process = await anyio.run_process(
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
        stdout=PIPE,
        stderr=PIPE,
        check=False,
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

    # Get the last speaker from the previous segment if it exists
    last_speaker = None
    if segment_index > 0 and interview.segments[segment_index - 1].utterances:
        last_speaker = interview.segments[segment_index - 1].utterances[-1].speaker

    # Parse the content into utterances
    utterances = []
    current_speaker = last_speaker or "S1"  # Default to S1 if no previous speaker
    speaker_pattern = re.compile(r"^\s*(\w+)\s*:\s*(.*)$")

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if match := speaker_pattern.match(line):
            # Line starts with a speaker tag
            current_speaker = match.group(1)
            text = match.group(2).strip()
        else:
            # Line continues with current speaker
            text = line

        if text:  # Only add if there's actual text content
            utterances.append(
                Utterance(
                    speaker=current_speaker,
                    text=text,
                )
            )

    # Update the segment
    segment.utterances = utterances
    INTERVIEWS.put(interview_id, interview)

    # Return the updated segment view
    with tag.div(classes="flex flex-wrap gap-4"):
        for i, utterance in enumerate(segment.utterances):
            render_utterance(interview_id, segment_index, i, utterance)


@app.post("/interview/{interview_id}/rename")
async def rename_interview(interview_id: str, new_name: str = Form(...)):
    """Processes the interview rename."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    interview.filename = new_name
    INTERVIEWS.put(interview_id, interview)

    return RedirectResponse(url=f"/interview/{interview_id}", status_code=303)


@app.get("/interview/{interview_id}/export")
async def export_interview(interview_id: str):
    """Export the interview as a DOCX file."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    # Create a new document
    doc = Document()

    # Add title
    title = doc.add_heading(interview.filename, level=1)
    title.runs[0].font.size = Pt(16)

    # Add each segment
    for segment in interview.segments:
        # Add timestamp as a subheading
        doc.add_heading(f"{segment.start_time} - {segment.end_time}", level=2).runs[
            0
        ].font.size = Pt(12)

        # Add utterances
        for utterance in segment.utterances:
            p = doc.add_paragraph()
            speaker_run = p.add_run(f"{utterance.speaker}: ")
            speaker_run.bold = True if utterance.speaker == "S1" else False
            text_run = p.add_run(utterance.text)
            text_run.bold = True if utterance.speaker == "S1" else False

    # Save to BytesIO
    docx_bytes = BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)

    # Return as downloadable file
    # Normalize filename: replace spaces with underscores and remove/replace any problematic characters
    safe_filename = re.sub(r"[^\w\-\.]", "_", interview.filename.replace(" ", "_"))
    # Ensure ASCII encoding for Content-Disposition header
    encoded_filename = safe_filename.encode("ascii", "ignore").decode("ascii")
    filename = f"{encoded_filename}.docx"

    return StreamingResponse(
        docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/interview/{interview_id}/context-segments")
async def update_context_segments(interview_id: str, context_segments: int = Form(...)):
    """Updates the number of context segments to use for transcription."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    interview.context_segments = max(
        0, min(5, context_segments)
    )  # Clamp between 0 and 5
    INTERVIEWS.put(interview_id, interview)

    return Response(status_code=204)  # No content response


@app.post("/interview/{interview_id}/segment/{segment_index}/update-speaker")
async def update_speaker(
    interview_id: str,
    segment_index: int,
    utterance_index: int = Form(...),
    key: str = Form(...),
):
    """Updates the speaker for a specific utterance."""
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    try:
        segment = interview.segments[segment_index]
        utterance = segment.utterances[utterance_index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Segment or utterance not found")

    # Update speaker if key is a digit
    if key.isdigit():
        utterance.speaker = f"S{key}"
        INTERVIEWS.put(interview_id, interview)

    # Return the updated utterance view
    return render_utterance(interview_id, segment_index, utterance_index, utterance)
