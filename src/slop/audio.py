import logging
import tempfile
from pathlib import Path
from subprocess import PIPE

import anyio
from starlette.exceptions import HTTPException

from slop.models import Part, Tape
from slop.promptflow import file as promptflow_file
from slop.store import get_blob, save_blob

logger = logging.getLogger(__name__)

AUDIO_MIME_TYPE = "audio/ogg"


def get_audio_bytes(audio_hash: str | None) -> bytes | None:
    """Return audio bytes for a blob hash if available."""
    if not audio_hash:
        return None
    if blob := get_blob(audio_hash):
        return blob[0]
    return None


async def insert_part_audio(
    tape: Tape,
    part: Part,
) -> str:
    """Ensure a part's audio exists and reference it lazily in the builder."""

    audio_hash = await _ensure_part_audio(tape, part)
    promptflow_file(f"blob:{audio_hash}", mime_type=AUDIO_MIME_TYPE)
    return audio_hash


async def _ensure_part_audio(
    tape: Tape,
    part: Part,
) -> str:
    if not tape.audio_hash:
        raise HTTPException(status_code=400, detail="Tape has no audio")

    if part.audio_hash:
        if get_blob(part.audio_hash):
            return part.audio_hash
        logger.warning(
            "Audio hash %s missing from blob store; re-extracting",
            part.audio_hash,
        )

    tape_blob = get_audio_bytes(tape.audio_hash)
    if tape_blob is None:
        raise HTTPException(status_code=404, detail="Tape audio blob not found")

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        temp_name = tmp.name
        tmp.write(tape_blob)
        tmp.flush()

    tmp_path = Path(temp_name)

    try:
        part_audio = await extract_part(
            tmp_path,
            part.start_time,
            part.end_time,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    part_hash = save_blob(part_audio, AUDIO_MIME_TYPE)
    part.audio_hash = part_hash
    return part_hash


async def extract_part(input_path: Path, start_time: str, end_time: str) -> bytes:
    """Extract a part from an audio file via ffmpeg."""
    logger.info("Extracting part from %s to %s", start_time, end_time)
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
