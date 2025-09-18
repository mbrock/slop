import logging
import re
import xml.etree.ElementTree as ET
from io import StringIO

from starlette.exceptions import HTTPException

from slop import gemini
from slop.audio import insert_segment_audio
from slop.gemini import GenerationConfig
from slop.models import (
    Interview,
    Segment,
    Utterance,
    get_interview,
)
from slop.promptflow import new_chat, tag, text
from slop.store import (
    ModelDecodeError,
    ModelNotFoundError,
)

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
    try:
        interview = get_interview(interview_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Interview not found") from exc
    except ModelDecodeError as exc:
        raise HTTPException(status_code=500, detail="Interview data invalid") from exc

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

    # Build a multi-turn conversation using promptflow helpers.
    with new_chat(upload=gemini.upload) as conversation:
        # Add each context segment as a separate conversation turn pair.
        if context_segments:
            for i, prev_segment in enumerate(context_segments):
                with conversation.user_turn():
                    if i == 0:
                        render_transcription_instruction()
                    with tag("audio", segment=str(i)):
                        await insert_segment_audio(interview, prev_segment)

                with conversation.model_turn():
                    with tag("transcript"):
                        for utterance in prev_segment.utterances:
                            with tag("sentence", speaker=utterance.speaker):
                                text(utterance.text, indent=True)

        with conversation.user_turn():
            if not context_segments:
                render_transcription_instruction()
            with tag("audio", segment="current"):
                await insert_segment_audio(interview, segment)

    # Request transcription; the model should now respond with a transcript only for the current audio.
    response = await conversation.complete(
        generation_config=GenerationConfig(temperature=0.1)
    )

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

    assert segment.audio_hash is not None
    return utterances, segment.audio_hash


async def improve_speaker_identification_segment(
    interview: Interview,
    segment: Segment,
    current_utterances: list[Utterance],
    context_segments: list[Segment] | None = None,
    hint: str | None = None,
) -> list[Utterance]:
    """
    Use Gemini to improve speaker identification for an audio segment.

    Args:
        interview: Interview containing the segment
        segment: The segment being relabeled
        current_utterances: Current utterances with speaker assignments
        context_segments: Optional list of previous segments for context
        hint: Optional user hint about the speakers

    Returns:
        List of utterances with improved speaker assignments
    """
    with new_chat(upload=gemini.upload) as prompt:
        with prompt.user_turn():
            # Include previous segments as context
            if context_segments:
                for i, prev_segment in enumerate(context_segments):
                    with tag("audio", segment=str(i)):
                        await insert_segment_audio(interview, prev_segment)

                    with tag("transcript", segment=str(i)):
                        for utterance in prev_segment.utterances:
                            with tag("sentence", speaker=utterance.speaker):
                                text(utterance.text, indent=True)

            await insert_segment_audio(interview, segment)

            with tag("transcript", segment="current"):
                for utterance in current_utterances:
                    with tag("sentence", speaker=utterance.speaker):
                        text(utterance.text, indent=True)

            render_speaker_instruction(hint)

    response = await prompt.complete(
        generation_config=GenerationConfig(temperature=0.1)
    )

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
