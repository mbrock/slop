import logging
import re
import xml.etree.ElementTree as ET
from io import StringIO

from starlette.exceptions import HTTPException

from slop import conf
from slop.audio import insert_part_audio
from slop.models import Part, Tape, Utterance, get_tape
from slop.promptflow import (
    from_model,
    from_user,
    generate,
    new_chat,
    tag,
    text,
)
from slop.store import ModelDecodeError, ModelNotFoundError

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
                    "Use speaker IDs like S1, S2, etc. Ensure S1 is the tapeer.",
                    "Keep natural sentence breaks - a sentence can be a single word or exclamation.",
                    "Output only valid XML with no extra text.",
                    "Include the transcript element.",
                    (
                        "The content is vetted for safety. You may omit prohibited content "
                        "but it shouldn't be necessary for these ethnographic tapes."
                    ),
                ),
                start=1,
            ):
                with tag("item", number=str(index)):
                    text(guideline)


def render_speaker_instruction(hint: str | None) -> None:
    """Emit markup guiding the speaker identification improvement request."""
    with tag("instruction"):
        text("Please improve the speaker identification in this part.")
        if hint:
            with tag("hint"):
                text(hint)

        text("Format your response as XML with the following structure:")
        with tag("example"):
            with tag("transcript", part="current"):
                with tag("sentence", speaker="S1"):
                    text("Hello, how are you?")
                with tag("sentence", speaker="S2"):
                    text("I'm doing great, thanks.")

        with tag("guidelines"):
            for index, guideline in enumerate(
                (
                    "Use speaker IDs like S1, S2, etc.",
                    "Consider voice characteristics and context from previous parts.",
                    "Be intelligent about speaker identification based on the context and conversation.",
                    "Include the transcript element with 'part=\"current\"' attribute.",
                ),
                start=1,
            ):
                with tag("item", number=str(index)):
                    text(guideline)


async def transcribe_audio_part(
    tape_id: str,
    start_time: str,
    end_time: str,
    context_parts: list[Part] | None = None,
) -> tuple[list[Utterance], str]:
    """
    Transcribe an audio part using Gemini and return utterances plus audio hash.
    """
    try:
        tape = get_tape(tape_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Tape not found") from exc
    except ModelDecodeError as exc:
        raise HTTPException(status_code=500, detail="Tape data invalid") from exc

    if not tape.audio_hash:
        raise HTTPException(status_code=400, detail="Tape has no audio")

    part = None
    for candidate in tape.parts:
        if candidate.start_time == start_time and candidate.end_time == end_time:
            part = candidate
            break

    if part is None:
        part = Part(start_time=start_time, end_time=end_time)

    with new_chat():
        if context_parts:
            for i, previous in enumerate(context_parts):
                with from_user():
                    if i == 0:
                        render_transcription_instruction()
                    with tag("audio", part=str(i)):
                        await insert_part_audio(tape, previous)

                with from_model():
                    with tag("transcript"):
                        for utterance in previous.utterances:
                            with tag("sentence", speaker=utterance.speaker):
                                text(utterance.text, indent=True)

        with from_user():
            if not context_parts:
                render_transcription_instruction()
            with tag("audio", part="current"):
                await insert_part_audio(tape, part)

        with conf.temperature(0.1):
            response = await generate()

    if not response.candidates:
        raise HTTPException(status_code=500, detail="Failed to transcribe part")

    full_text = response.candidates[0].content.parts[0].text
    xml_match = re.search(r"<transcript>(.*?)</transcript>", full_text, re.DOTALL)
    if not xml_match:
        raise HTTPException(status_code=500, detail="Failed to find transcription XML")

    xml_text = xml_match.group(0)
    utterances = parse_transcription_xml(xml_text)
    if not utterances:
        raise HTTPException(status_code=500, detail="Failed to parse transcription XML")

    assert part.audio_hash is not None
    return utterances, part.audio_hash


async def improve_speaker_identification_part(
    tape: Tape,
    part: Part,
    current_utterances: list[Utterance],
    context_parts: list[Part] | None = None,
    hint: str | None = None,
) -> list[Utterance]:
    """Use Gemini to improve speaker tags for a part."""
    with new_chat():
        with from_user():
            if context_parts:
                for i, previous in enumerate(context_parts):
                    with tag("audio", part=str(i)):
                        await insert_part_audio(tape, previous)

                    with tag("transcript", part=str(i)):
                        for utterance in previous.utterances:
                            with tag("sentence", speaker=utterance.speaker):
                                text(utterance.text, indent=True)

            await insert_part_audio(tape, part)

            with tag("transcript", part="current"):
                for utterance in current_utterances:
                    with tag("sentence", speaker=utterance.speaker):
                        text(utterance.text, indent=True)

            render_speaker_instruction(hint)

        with conf.temperature(0.1):
            response = await generate()

    if not response.candidates:
        raise HTTPException(status_code=500, detail="Failed to improve speaker identification")

    full_text = response.candidates[0].content.parts[0].text
    xml_match = re.search(
        r"<transcript\s+part=\"current\">(.*?)</transcript>",
        full_text,
        re.DOTALL,
    )
    if not xml_match:
        raise HTTPException(status_code=500, detail="Failed to find transcription XML")

    xml_text = xml_match.group(0)
    utterances = parse_transcription_xml(xml_text)
    if not utterances:
        raise HTTPException(status_code=500, detail="Failed to parse transcription XML")

    return utterances


def parse_transcription_xml(xml_text: str) -> list[Utterance]:
    """Parse Gemini XML into Utterance objects."""
    try:
        tree = ET.parse(StringIO(xml_text))
        root = tree.getroot()
        utterances: list[Utterance] = []
        for sentence in root.findall("sentence"):
            speaker = sentence.get("speaker") or "UNK"
            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=sentence.text.strip() if sentence.text else "",
                )
            )
        return utterances
    except ET.ParseError as exc:  # pragma: no cover - narrow failure path
        logger.error("Failed to parse XML: %s", exc)
        logger.error("XML content was: %s", xml_text)
        return []

