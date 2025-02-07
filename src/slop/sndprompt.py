import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

from fastapi import HTTPException
import rich
import trio

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


async def transcribe_audio_segment(
    interview_id: str,
    start_time: str,
    end_time: str,
    context_segments: list[Segment] | None = None,
) -> tuple[list[Utterance], str]:
    """
    Transcribe an audio segment using Gemini.
    Returns the transcribed utterances.

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

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        # Get the full interview audio
        audio_data, _ = BLOBS.get(interview.audio_hash)
        tmp.write(audio_data)
        tmp.flush()
        tmp_path = Path(tmp.name)

        try:
            # Extract the segment audio
            segment_audio = await extract_segment(tmp_path, start_time, end_time)
            segment_hash = BLOBS.put(segment_audio, "audio/ogg")

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
                        parts.append(Part(text=f'<audio id="{i}">'))
                        parts.append(
                            Part(
                                fileData=FileData(
                                    fileUri=prev_file.uri, mimeType="audio/ogg"
                                )
                            )
                        )
                        parts.append(Part(text="</audio>"))
                        context_xml = (
                            f'<transcript id="{i}">'
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
                    Part(text='<audio id="current">'),
                    Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                    Part(text="</audio>"),
                    Part(
                        text="""Format your response as XML with the following structure:

<transcript id="current">
  <sentence speaker="S1">Hello, how are you?</sentence>
  <sentence speaker="S2">I'm doing great, thanks.</sentence>
</transcript>

Guidelines:
1. Use speaker IDs like S1, S2, etc. Ensure S1 is the interviewer.
2. Keep natural sentence breaks - a sentence can be a single word or exclamation.
3. Consider voice characteristics and context from previous segments.
4. Output only valid XML with no extra text.
5. Include the transcript element with 'id="current"' attribute.
"""
                    ),
                ]
            )

            # Request transcription
            request = GenerateRequest(
                contents=[Content(role="user", parts=parts)],
                generationConfig=GenerationConfig(temperature=0.3),
            )
            response = await client.generate_content_sync(
                request, model=interview.model_name
            )

            if not response.candidates:
                raise HTTPException(
                    status_code=500, detail="Failed to transcribe segment"
                )

            # Extract the text content
            full_text = response.candidates[0].content.parts[0].text

            # Find the XML with segment attribute
            xml_match = re.search(
                r"<transcript\s+id=\"current\">(.*?)</transcript>",
                full_text,
                re.DOTALL,
            )
            if not xml_match:
                raise HTTPException(
                    status_code=500, detail="Failed to find transcription XML"
                )

            xml_text = xml_match.group(0)
            utterances = parse_transcription_xml(xml_text)
            if not utterances:
                raise HTTPException(
                    status_code=500, detail="Failed to parse transcription XML"
                )

            return utterances, segment_hash
        finally:
            tmp_path.unlink()
            await client.delete_file(file.name)


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

    try:
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
                    parts.append(Part(text=f'<audio id="{i}">'))
                    parts.append(
                        Part(
                            fileData=FileData(
                                fileUri=prev_file.uri, mimeType="audio/ogg"
                            )
                        )
                    )
                    parts.append(Part(text="</audio>"))
                    parts.append(Part(text=f'<transcript id="{i}">'))
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
                Part(
                    text="Current segment that needs speaker identification improvement:"
                ),
                Part(text='<audio id="current">'),
                Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                Part(text="</audio>"),
                Part(
                    text='<transcript id="current">\n'
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

<transcript id="current">
  <sentence speaker="S1">Hello, how are you?</sentence>
  <sentence speaker="S2">I'm doing great, thanks.</sentence>
</transcript>

Guidelines:
1. Use speaker IDs like S1, S2, etc. Ensure S1 is the interviewer.
2. Keep natural sentence breaks - a sentence can be a single word or exclamation.
3. Consider voice characteristics and context from previous segments.
4. Output only valid XML with no extra text.
5. Include the transcript element with 'id="current"' attribute.
"""
                ),
            ]
        )

        # Request improved speaker identification
        request = GenerateRequest(
            contents=[Content(role="user", parts=parts)],
            generationConfig=GenerationConfig(temperature=0.4),
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
            r"<transcript\s+id=\"current\">(.*?)</transcript>", full_text, re.DOTALL
        )
        if not xml_match:
            raise HTTPException(
                status_code=500, detail="Failed to find transcription XML"
            )

        xml_text = xml_match.group(0)
        utterances = parse_transcription_xml(xml_text)
        if not utterances:
            raise HTTPException(
                status_code=500, detail="Failed to parse transcription XML"
            )

        return utterances

    finally:
        await client.delete_file(file.name)


def parse_transcription_xml(xml_text: str) -> list[Utterance]:
    """
    Parse transcription XML into a list of Utterances.
    Expected format:
        <transcript id="current">
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
