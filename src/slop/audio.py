import logging
import tempfile
from pathlib import Path
from subprocess import PIPE

import anyio
from fastapi import HTTPException

from slop import gemini
from slop.gemini import File
from slop.models import Interview, Segment, get_blob, save_blob
from slop.promptflow import file as promptflow_file

logger = logging.getLogger(__name__)

AUDIO_MIME_TYPE = "audio/ogg"


def get_audio_bytes(audio_hash: str | None) -> bytes | None:
    """Return audio bytes for a blob hash if available."""
    if not audio_hash:
        return None
    if blob := get_blob(audio_hash):
        return blob[0]
    return None


async def insert_audio_bytes(data: bytes) -> File:
    """Upload raw audio bytes and insert them into the active promptflow builder."""
    uploaded = await gemini.upload_bytes(data, mime_type=AUDIO_MIME_TYPE)
    promptflow_file(uploaded, mime_type=AUDIO_MIME_TYPE)
    return uploaded


async def insert_segment_audio(
    interview: Interview,
    segment: Segment,
) -> str:
    """Ensure, upload, and insert a segment's audio in the active builder."""

    audio_bytes, audio_hash = await _ensure_segment_audio(interview, segment)
    segment.audio_hash = audio_hash
    await insert_audio_bytes(audio_bytes)
    return audio_hash


async def _ensure_segment_audio(
    interview: Interview,
    segment: Segment,
) -> tuple[bytes, str]:
    if not interview.audio_hash:
        raise HTTPException(status_code=400, detail="Interview has no audio")

    if segment.audio_hash:
        if cached := get_audio_bytes(segment.audio_hash):
            return cached, segment.audio_hash
        logger.warning(
            "Audio hash %s missing from blob store; re-extracting",
            segment.audio_hash,
        )

    interview_blob = get_audio_bytes(interview.audio_hash)
    if interview_blob is None:
        raise HTTPException(status_code=404, detail="Interview audio blob not found")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        temp_name = tmp.name
        tmp.write(interview_blob)
        tmp.flush()

    tmp_path = Path(temp_name)

    try:
        segment_audio = await extract_segment(
            tmp_path,
            segment.start_time,
            segment.end_time,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    segment_hash = save_blob(segment_audio, AUDIO_MIME_TYPE)
    segment.audio_hash = segment_hash
    return segment_audio, segment_hash


async def extract_segment(input_path: Path, start_time: str, end_time: str) -> bytes:
    """Extract a segment from an audio file via ffmpeg."""
    logger.info("Extracting segment from %s to %s", start_time, end_time)
    process = await anyio.run_process(
        [
            "ffmpeg",
            "-flags",
            "+bitexact",
            "-i",
            str(input_path),
            "-ss",
            start_time,
            "-to",
            end_time,
            "-c:a",
            "libvorbis",
            "-q:a",
            "8",
            "-f",
            "ogg",
            "pipe:1",
        ],
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    return process.stdout
