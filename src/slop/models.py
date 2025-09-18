import os
from pathlib import Path

from pydantic import BaseModel, Field

from slop.store import find, save
from slop.store import list_models as store_list_models


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
    audio_hash: str | None = None  # Hash of the processed audio file
    duration: str = Field(
        default="00:00:00",
        description="Total duration of the interview (HH:MM:SS)",
    )
    current_position: str = Field(
        default="00:00:00",
        description="Current position in the interview (HH:MM:SS)",
    )
    context_segments: int = Field(
        default=1,
        description="Number of previous segments to include as context for transcription",
    )
    segments: list[Segment] = []


def get_data_dir() -> Path:
    """Get the data directory from environment or default."""
    return Path(os.getenv("IEVA_DATA", "/data"))


def get_interview(interview_id: str) -> Interview:
    """Compatibility wrapper for loading :class:`Interview` instances."""

    return find(Interview, interview_id)


def save_interview(interview: Interview) -> str:
    """Compatibility wrapper for storing :class:`Interview` instances."""

    return save(interview)


def list_interviews() -> list[Interview]:
    """Compatibility wrapper returning every stored interview."""

    return store_list_models(Interview)
