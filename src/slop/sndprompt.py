import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

from fastapi import HTTPException
import rich
import trio

from slop.gemini import Content, FileData, GeminiClient, GenerateRequest, Part
from slop.models import BLOBS, INTERVIEWS, Segment, Utterance

logger = logging.getLogger(__name__)


async def transcribe_audio_segment(
    interview_id: str,
    start_time: str,
    end_time: str,
    context_segments: list[Segment] | None = None,
) -> list[Utterance]:
    """
    Transcribe an audio segment using Gemini.
    Returns the transcribed utterances.

    Args:
        interview_id: The ID of the interview
        start_time: Start time in HH:MM:SS format
        end_time: End time in HH:MM:SS format
        context_segments: List of previous segments to use as context, ordered from oldest to newest
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
                        parts.extend(
                            [
                                Part(
                                    text=f"Context segment {i + 1} ({prev_segment.start_time} - {prev_segment.end_time}):"
                                ),
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
            parts.extend(
                [
                    Part(text="New audio segment:"),
                    Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                    Part(
                        text="""Format your response as XML with the following structure:

<transcript segment="{SEGMENT_NUMBER}">
  <pair index="1">
    <utterance speaker="S1">Hello, how are you?</utterance>
  </pair>
  <pair index="2">
    <utterance speaker="S2">I'm doing great, thanks.</utterance>
  </pair>
</transcript>

Instructions:
1. Only transcribe the new audio provided in this request (ignore context segments).
2. Wrap each audio/text pair in a <pair> element with a unique index attribute.
3. Use speaker IDs like S1, S2, etc.
4. Output only valid XML with no extra text.
5. Ensure the transcript element includes a 'segment' attribute corresponding to the current segment number.
"""
                    ),
                ]
            )

            # Request transcription
            request = GenerateRequest(contents=[Content(role="user", parts=parts)])
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
) -> list[Utterance]:
    """
    Use Gemini to improve speaker identification for an audio segment.

    Args:
        segment_audio: The audio data for the segment
        current_utterances: Current utterances with speaker assignments
        context_segments: Optional list of previous segments for context
        hint: Optional user hint about the speakers

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
                    parts.extend(
                        [
                            Part(
                                text=f"Context segment {i + 1} ({prev_segment.start_time} - {prev_segment.end_time}):"
                            ),
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

        # Add current segment with its current transcription
        parts.extend(
            [
                Part(
                    text="Current segment that needs speaker identification improvement:"
                ),
                Part(fileData=FileData(fileUri=file.uri, mimeType="audio/ogg")),
                Part(
                    text="Current transcription:\n"
                    + "".join(
                        f'<utterance speaker="{u.speaker}">{u.text}</utterance>'
                        for u in current_utterances
                    )
                ),
                Part(
                    text=f"""Please improve the speaker identification in this segment.
{f"User hint about the speakers: {hint}" if hint else ""}

Format your response as XML with the following structure:

<transcript segment="current">
  <pair index="1">
    <utterance speaker="S1">Hello, how are you?</utterance>
  </pair>
  <pair index="2">
    <utterance speaker="S2">I'm doing great, thanks.</utterance>
  </pair>
</transcript>

Guidelines:
1. Use the provided transcription as a base and only output corrected speaker assignments.
2. Wrap each utterance in a <pair> element with a unique index attribute.
3. Consider voice characteristics and context; ensure S1 is the interviewer.
4. Output only valid XML with no extra text.
"""
                ),
            ]
        )

        # Request improved speaker identification
        request = GenerateRequest(contents=[Content(role="user", parts=parts)])
        response = await client.generate_content_sync(request)
        if not response.candidates:
            raise HTTPException(
                status_code=500, detail="Failed to improve speaker identification"
            )

        # Extract the text content
        full_text = response.candidates[0].content.parts[0].text

        # Find the XML
        xml_match = re.search(r"<transcript>(.*?)</transcript>", full_text, re.DOTALL)
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
        <transcript>
            <utterance speaker="S1">Hello</utterance>
            ...
        </transcript>
    """
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
