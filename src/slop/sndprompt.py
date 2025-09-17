import hashlib
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from subprocess import PIPE

from fastapi import HTTPException
import anyio
import rich

from slop.gemini import (
    Content,
    FileData,
    GeminiClient,
    GenerateRequest,
    GenerationConfig,
    Part,
)
from slop.models import BLOBS, INTERVIEWS, Segment, Utterance

logger = logging.getLogger(__name__)


TRANSCRIPTION_INSTRUCTIONS = """Format your response as XML with the following structure:

<transcript>
  <sentence speaker="S1">Hello, how are you?</sentence>
  <sentence speaker="S2">I'm doing great, thanks.</sentence>
</transcript>

Guidelines:
1. Use speaker IDs like S1, S2, etc. Ensure S1 is the interviewer.
2. Keep natural sentence breaks - a sentence can be a single word or exclamation.
3. Output only valid XML with no extra text.
4. Include the transcript element.
5. The content is vetted for safety. You may omit prohibited content but it shouldn't be necessary for these ethnographic interviews.
"""


async def transcribe_audio_segment_v1(
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
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    if not interview.audio_hash:
        raise HTTPException(status_code=400, detail="Interview has no audio")

    # Find or create segment
    segment = None
    for s in interview.segments:
        if s.start_time == start_time and s.end_time == end_time:
            segment = s
            break

    if not segment:
        segment = Segment(start_time=start_time, end_time=end_time)

    # Get or extract the audio segment
    segment_audio = None
    if segment.audio_hash:
        if stored_segment := BLOBS.get(segment.audio_hash):
            logger.info(f"Found cached audio segment {segment.audio_hash}")
            segment_audio = stored_segment[0]

    if not segment_audio:
        # Extract the segment if not cached
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Get the full interview audio
            audio_data, _ = BLOBS.get(interview.audio_hash)
            tmp.write(audio_data)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                # Extract the segment audio
                segment_audio = await extract_segment(tmp_path, start_time, end_time)
                # Store in blob store and update segment
                segment.audio_hash = BLOBS.put(segment_audio, "audio/ogg")
                logger.info(
                    f"Extracted and stored new audio segment {segment.audio_hash}"
                )
            finally:
                tmp_path.unlink()

    # Upload segment to Gemini
    client = GeminiClient()
    file = await client.upload_bytes(
        segment_audio,
        mime_type="audio/ogg",
        display_name=f"segment_{start_time}",
    )

    parts = []

    # Include previous segments as context if available
    if context_segments:
        for i, prev_segment in enumerate(context_segments):
            if prev_segment.audio_hash:
                prev_audio_data, _ = BLOBS.get(prev_segment.audio_hash)
                prev_file = await client.upload_bytes(
                    prev_audio_data,
                    mime_type="audio/ogg",
                    display_name=f"context_segment_{i}_{prev_segment.start_time}",
                )
                parts.append(Part(text=f'<audio segment="{i}">'))
                parts.append(
                    Part(fileData=FileData(fileUri=prev_file.uri, mimeType="audio/ogg"))
                )
                parts.append(Part(text="</audio>"))
                context_xml = (
                    f'<transcript segment="{i}">'
                    + "".join(
                        f'<sentence speaker="{u.speaker}">{u.text}</sentence>'
                        for u in prev_segment.utterances
                    )
                    + "</transcript>"
                )
                parts.append(Part(text=context_xml))

    # Add current segment's audio and instructions
    parts.extend(
        [
            Part(text='<audio segment="current">'),
            Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
            Part(text="</audio>"),
            Part(
                text="""Format your response as XML with the following structure:

<transcript segment="current">
  <sentence speaker="S1">Hello, how are you?</sentence>
  <sentence speaker="S2">I'm doing great, thanks.</sentence>
</transcript>

Guidelines:
1. Use speaker IDs like S1, S2, etc. Ensure S1 is the interviewer.
2. Keep natural sentence breaks - a sentence can be a single word or exclamation.
3. Output only valid XML with no extra text.
4. Include the transcript element with 'segment="current"' attribute.
"""
            ),
        ]
    )

    # Request transcription
    request = GenerateRequest(
        contents=[Content(role="user", parts=parts)],
        generationConfig=GenerationConfig(temperature=0.1),
    )
    response = await client.generate_content_sync(request, model=interview.model_name)

    if not response.candidates:
        raise HTTPException(status_code=500, detail="Failed to transcribe segment")

    # Extract the text content
    full_text = response.candidates[0].content.parts[0].text

    # Find the XML with segment attribute
    xml_match = re.search(
        r"<transcript\s+segment=\"current\">(.*?)</transcript>",
        full_text,
        re.DOTALL,
    )
    if not xml_match:
        raise HTTPException(status_code=500, detail="Failed to find transcription XML")

    xml_text = xml_match.group(0)
    utterances = parse_transcription_xml(xml_text)
    if not utterances:
        raise HTTPException(status_code=500, detail="Failed to parse transcription XML")

    return utterances, segment.audio_hash

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
    if not (interview := INTERVIEWS.get(interview_id)):
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
    segment_audio = None
    if segment.audio_hash:
        if stored_segment := BLOBS.get(segment.audio_hash):
            logger.info(f"Found cached audio segment {segment.audio_hash}")
            segment_audio = stored_segment[0]

    if not segment_audio:
        # Extract the segment if not cached.
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Get the full interview audio.
            audio_data, _ = BLOBS.get(interview.audio_hash)
            tmp.write(audio_data)
            tmp.flush()
            tmp_path = Path(tmp.name)

            try:
                # Extract the segment audio.
                segment_audio = await extract_segment(tmp_path, start_time, end_time)
                # Store in blob store and update segment.
                segment.audio_hash = BLOBS.put(segment_audio, "audio/ogg")
                logger.info(f"Extracted and stored new audio segment {segment.audio_hash}")
            finally:
                tmp_path.unlink()

    # Upload the current segment audio to Gemini.
    client = GeminiClient()
    current_file = await client.upload_bytes(
        segment_audio,
        mime_type="audio/ogg",
        display_name=f"segment_{start_time}",
    )

    # Build a multi-turn conversation.
    conversation: list[Content] = []

    # Add each context segment as a separate conversation turn pair.
    if context_segments:
        for i, prev_segment in enumerate(context_segments):
            if prev_segment.audio_hash:
                prev_audio_data, _ = BLOBS.get(prev_segment.audio_hash)
                prev_file = await client.upload_bytes(
                    prev_audio_data,
                    mime_type="audio/ogg",
                    display_name=f"context_segment_{i}_{prev_segment.start_time}",
                )
                # For the very first turn, include transcription instructions.
                if i == 0:
                    conversation.append(
                        Content(
                            role="user",
                            parts=[
                                Part(text=TRANSCRIPTION_INSTRUCTIONS),
                                Part(text="<audio>"),
                                Part(
                                    fileData=FileData(
                                        fileUri=prev_file.uri, mimeType="audio/ogg"
                                    )
                                ),
                                Part(text="</audio>"),
                            ],
                        )
                    )
                else:
                    conversation.append(
                        Content(
                            role="user",
                            parts=[
                                Part(text="<audio>"),
                                Part(
                                    fileData=FileData(
                                        fileUri=prev_file.uri, mimeType="audio/ogg"
                                    )
                                ),
                                Part(text="</audio>"),
                            ],
                        )
                    )

                # Now include the known transcript as the assistant's reply.
                transcript_xml = (
                    "<transcript>"
                    + "".join(
                        f'<sentence speaker="{u.speaker}">{u.text}</sentence>'
                        for u in prev_segment.utterances
                    )
                    + "</transcript>"
                )
                conversation.append(
                    Content(role="model", parts=[Part(text=transcript_xml)])
                )

    # Finally, add the current segment as the latest user turn.
    conversation.append(
        Content(
            role="user",
            parts=[
                # Add transcription instructions if this is the first and only turn
                *([] if context_segments else [Part(text=TRANSCRIPTION_INSTRUCTIONS)]),
                Part(text="<audio>"),
                Part(
                    fileData=FileData(
                        fileUri=current_file.uri, mimeType="audio/ogg"
                    )
                ),
                Part(text="</audio>"),
            ],
        )
    )

    # Request transcription; the model should now respond with a transcript only for the current audio.
    request = GenerateRequest(
        contents=conversation,
        generationConfig=GenerationConfig(temperature=0.1),
    )
    response = await client.generate_content_sync(request, model=interview.model_name)

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

    return utterances, segment.audio_hash

async def improve_speaker_identification_segment(
    segment_audio: bytes,
    current_utterances: list[Utterance],
    context_segments: list[Segment] | None = None,
    hint: str | None = None,
    model_name: str = "gemini-2.0-flash-exp",
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
    client = GeminiClient()
    file = await client.upload_bytes(
        segment_audio,
        mime_type="audio/ogg",
        display_name="segment_to_improve",
    )

    parts = []

    # Include previous segments as context
    if context_segments:
        for i, prev_segment in enumerate(context_segments):
            if prev_segment.audio_hash:
                prev_audio_data, _ = BLOBS.get(prev_segment.audio_hash)
                prev_file = await client.upload_bytes(
                    prev_audio_data,
                    mime_type="audio/ogg",
                    display_name=f"context_segment_{i}_{prev_segment.start_time}",
                )
                parts.append(Part(text=f'<audio segment="{i}">'))
                parts.append(
                    Part(fileData=FileData(fileUri=prev_file.uri, mimeType="audio/ogg"))
                )
                parts.append(Part(text="</audio>"))
                parts.append(Part(text=f'<transcript segment="{i}">'))
                parts.append(
                    Part(
                        text="".join(
                            f'<sentence speaker="{u.speaker}">{u.text}</sentence>'
                            for u in prev_segment.utterances
                        )
                    )
                )
                parts.append(Part(text="</transcript>"))

    # Add current segment with its current transcription
    parts.extend(
        [
            Part(text='<audio segment="current">'),
            Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
            Part(text="</audio>"),
            Part(
                text='<transcript segment="current">\n'
                + "".join(
                    f'<sentence speaker="{u.speaker}">{u.text}</sentence>'
                    for u in current_utterances
                )
                + "\n</transcript>"
            ),
            Part(
                text=f"""Please improve the speaker identification in this segment.
{f"User hint about the speakers: {hint}" if hint else ""}

Format your response as XML with the following structure:

<transcript segment="current">
<sentence speaker="S1">Hello, how are you?</sentence>
<sentence speaker="S2">I'm doing great, thanks.</sentence>
</transcript>

Guidelines:
1. Use speaker IDs like S1, S2, etc.
2. Consider voice characteristics and context from previous segments.
3. Be intelligent about speaker identification based on the context and conversation.
4. Include the transcript element with 'segment="current"' attribute.
"""
            ),
        ]
    )

    # Request improved speaker identification
    request = GenerateRequest(
        contents=[Content(role="user", parts=parts)],
        generationConfig=GenerationConfig(temperature=0.1),
    )
    response = await client.generate_content_sync(request, model=model_name)
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
            utterances.append(
                Utterance(
                    speaker=sent.get("speaker"),
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
