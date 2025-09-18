import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from subprocess import PIPE

import anyio
from fastapi import HTTPException

from slop import gemini
from slop.gemini import GenerateRequest, GenerationConfig
from slop.promptflow import ConversationBuilder, file, tag, text
from slop.models import get_blob, save_blob, get_interview, Segment, Utterance

logger = logging.getLogger(__name__)


def render_transcription_instruction() -> None:
    """Emit markup describing the expected transcription output format."""
    with tag("instruction"):
        text("Format your response as XML with the following structure:")
        with tag("example"):
            with tag("transcript"):
                for speaker, sample_text in (
                    ("S1", "Hello, how are you?"),
                    ("S2", "I'm doing great, thanks."),
                ):
                    with tag("sentence", speaker=speaker):
                        text(sample_text)
        with tag("guidelines"):
            for index, guideline in enumerate(
                (
                    "Use speaker IDs like S1, S2, etc. Ensure S1 is the interviewer.",
                    (
                        "Keep natural sentence breaks - a sentence can be a single "
                        "word or exclamation."
                    ),
                    "Output only valid XML with no extra text.",
                    "Include the transcript element.",
                    (
                        "The content is vetted for safety. You may omit prohibited "
                        "content but it shouldn't be necessary for these "
                        "ethnographic interviews."
                    ),
                ),
                start=1,
            ):
                with tag("item", number=str(index)):
                    text(guideline)


def render_speaker_instruction(hint: str | None) -> None:
    """Emit markup guiding the speaker identification improvement request."""
    with tag("instruction"):
        text("Please improve the speaker identification in this segment.")
        if hint:
            with tag("hint"):
                text(hint)

        text("Format your response as XML with the following structure:")
        with tag("example"):
            with tag("transcript", segment="current"):
                with tag("sentence", speaker="S1"):
                    text("Hello, how are you?")
                with tag("sentence", speaker="S2"):
                    text("I'm doing great, thanks.")

        with tag("guidelines"):
            for index, guideline in enumerate(
                (
                    "Use speaker IDs like S1, S2, etc.",
                    "Consider voice characteristics and context from previous segments.",
                    (
                        "Be intelligent about speaker identification based on the "
                        "context and conversation."
                    ),
                    "Include the transcript element with 'segment=\"current\"' attribute.",
                ),
                start=1,
            ):
                with tag("item", number=str(index)):
                    text(guideline)


async def transcribe_audio_segment(
    interview_id: str,
    start_time: str,
    end_time: str,
    context_segments: list[Segment] | None = None,
) -> tuple[list[Utterance], str]:
    """
    Transcribe an audio segment using Gemini.
    Returns the transcribed utterances and the segment's audio hash.

    Args:
        interview_id: The ID of the interview
        start_time: Start time in HH:MM:SS format
        end_time: End time in HH:MM:SS format
        context_segments: List of previous segments for context, ordered from oldest to newest
    """
    # Retrieve interview and validate.
    if not (interview := get_interview(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    if not interview.audio_hash:
        raise HTTPException(status_code=400, detail="Interview has no audio")

    # Find or create segment.
    segment = None
    for s in interview.segments:
        if s.start_time == start_time and s.end_time == end_time:
            segment = s
            break

    if not segment:
        segment = Segment(start_time=start_time, end_time=end_time)

    # Get or extract the audio segment.
    segment_audio: bytes | None = None
    if segment.audio_hash:
        if stored_segment := get_blob(segment.audio_hash):
            logger.info(f"Found cached audio segment {segment.audio_hash}")
            segment_audio = stored_segment[0]

    if not segment_audio:
        # Extract the segment if not cached.
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Get the full interview audio.
            interview_blob = get_blob(interview.audio_hash)
            if not interview_blob:
                raise HTTPException(
                    status_code=404,
                    detail="Interview audio blob not found",
                )
            audio_data, _ = interview_blob
            tmp.write(audio_data)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                # Extract the segment audio.
                segment_audio = await extract_segment(tmp_path, start_time, end_time)
                # Store in blob store and update segment.
                segment.audio_hash = save_blob(segment_audio, "audio/ogg")
                logger.info(
                    f"Extracted and stored new audio segment {segment.audio_hash}"
                )
            finally:
                tmp_path.unlink()

    if segment_audio is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to load audio segment",
        )

    # Upload the current segment audio to Gemini.
    current_file = await gemini.upload_bytes(
        segment_audio,
        mime_type="audio/ogg",
        display_name=f"segment_{start_time}",
    )

    # Build a multi-turn conversation using promptflow helpers.
    conversation = ConversationBuilder()

    # Add each context segment as a separate conversation turn pair.
    if context_segments:
        for i, prev_segment in enumerate(context_segments):
            if not prev_segment.audio_hash:
                continue

            prev_blob = get_blob(prev_segment.audio_hash)
            if not prev_blob:
                logger.warning(
                    "Missing audio blob for previous segment %s",
                    prev_segment.audio_hash,
                )
                continue

            prev_audio_data, _ = prev_blob
            prev_file = await gemini.upload_bytes(
                prev_audio_data,
                mime_type="audio/ogg",
                display_name=f"context_segment_{i}_{prev_segment.start_time}",
            )

            with conversation.user_turn():
                if i == 0:
                    render_transcription_instruction()
                with tag("audio"):
                    file(prev_file, mime_type="audio/ogg")

            with conversation.model_turn():
                with tag("transcript"):
                    for utterance in prev_segment.utterances:
                        with tag("sentence", speaker=utterance.speaker):
                            text(utterance.text, indent=True)

    # Finally, add the current segment as the latest user turn.
    with conversation.user_turn():
        if not context_segments:
            render_transcription_instruction()
        with tag("audio"):
            file(current_file, mime_type="audio/ogg")

    # Request transcription; the model should now respond with a transcript only for the current audio.
    request = GenerateRequest(
        contents=conversation.to_contents(),
        generationConfig=GenerationConfig(temperature=0.1),
    )
    response = await gemini.generate_content_sync(request)

    if not response.candidates:
        raise HTTPException(status_code=500, detail="Failed to transcribe segment")

    # The final assistant turn (the generated response) should contain the <transcript> XML.
    full_text = response.candidates[0].content.parts[0].text

    # Search for the transcript XML.
    xml_match = re.search(r"<transcript>(.*?)</transcript>", full_text, re.DOTALL)
    if not xml_match:
        raise HTTPException(status_code=500, detail="Failed to find transcription XML")

    xml_text = xml_match.group(0)
    utterances = parse_transcription_xml(xml_text)
    if not utterances:
        raise HTTPException(status_code=500, detail="Failed to parse transcription XML")

    if not segment.audio_hash:
        raise HTTPException(
            status_code=500,
            detail="Audio hash missing for segment",
        )

    return utterances, segment.audio_hash


async def improve_speaker_identification_segment(
    segment_audio: bytes,
    current_utterances: list[Utterance],
    context_segments: list[Segment] | None = None,
    hint: str | None = None,
) -> list[Utterance]:
    """
    Use Gemini to improve speaker identification for an audio segment.

    Args:
        segment_audio: The audio data for the segment
        current_utterances: Current utterances with speaker assignments
        context_segments: Optional list of previous segments for context
        hint: Optional user hint about the speakers
        model_name: The name of the model to use for improved speaker identification

    Returns:
        List of utterances with improved speaker assignments
    """
    segment_file = await gemini.upload_bytes(
        segment_audio,
        mime_type="audio/ogg",
        display_name="segment_to_improve",
    )

    prompt = ConversationBuilder()

    with prompt.user_turn():
        # Include previous segments as context
        if context_segments:
            for i, prev_segment in enumerate(context_segments):
                if not prev_segment.audio_hash:
                    continue

                prev_blob = get_blob(prev_segment.audio_hash)
                if not prev_blob:
                    logger.warning(
                        "Missing audio blob for previous segment %s",
                        prev_segment.audio_hash,
                    )
                    continue

                prev_audio_data, _ = prev_blob
                prev_file = await gemini.upload_bytes(
                    prev_audio_data,
                    mime_type="audio/ogg",
                    display_name=f"context_segment_{i}_{prev_segment.start_time}",
                )

                with tag("audio", segment=str(i)):
                    file(prev_file, mime_type="audio/ogg")

                with tag("transcript", segment=str(i)):
                    for utterance in prev_segment.utterances:
                        with tag("sentence", speaker=utterance.speaker):
                            text(utterance.text, indent=True)

        # Add current segment with its current transcription
        with tag("audio", segment="current"):
            file(segment_file, mime_type="audio/ogg")

        with tag("transcript", segment="current"):
            for utterance in current_utterances:
                with tag("sentence", speaker=utterance.speaker):
                    text(utterance.text, indent=True)

        render_speaker_instruction(hint)

    # Request improved speaker identification
    request = GenerateRequest(
        contents=prompt.to_contents(),
        generationConfig=GenerationConfig(temperature=0.1),
    )
    response = await gemini.generate_content_sync(request)

    if not response.candidates:
        raise HTTPException(
            status_code=500, detail="Failed to improve speaker identification"
        )

    # Extract the text content
    full_text = response.candidates[0].content.parts[0].text

    # Find the XML with segment attribute
    xml_match = re.search(
        r"<transcript\s+segment=\"current\">(.*?)</transcript>", full_text, re.DOTALL
    )
    if not xml_match:
        raise HTTPException(status_code=500, detail="Failed to find transcription XML")

    xml_text = xml_match.group(0)
    utterances = parse_transcription_xml(xml_text)
    if not utterances:
        raise HTTPException(status_code=500, detail="Failed to parse transcription XML")

    return utterances


def parse_transcription_xml(xml_text: str) -> list[Utterance]:
    """
    Parse transcription XML into a list of Utterances.
    Expected format:
        <transcript segment="current">
            <sentence speaker="S1">Hello</sentence>
            ...
        </transcript>
    """
    try:
        tree = ET.parse(StringIO(xml_text))
        root = tree.getroot()
        utterances = []
        for sent in root.findall("sentence"):
            speaker = sent.get("speaker") or "UNK"
            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=sent.text.strip() if sent.text else "",
                )
            )
        return utterances
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        logger.error(f"XML content was: {xml_text}")
        return []


async def extract_segment(input_path: Path, start_time: str, end_time: str) -> bytes:
    """
    Extract a segment from an audio file using ffmpeg.
    Returns the extracted segment as bytes.
    """
    logger.info(f"Extracting segment from {start_time} to {end_time}")

    process = await anyio.run_process(
        [
            "ffmpeg",
            "-flags",
            "+bitexact",
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
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    return process.stdout
