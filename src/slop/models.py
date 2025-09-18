"""Domain models for the tape transcription workflow."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from slop.store import find, list_models as store_list_models, save


class PartitionPart(BaseModel):
    """A proposed tape partition returned by pre-processing."""

    start_time: str = Field(description="Start time of the part (HH:MM:SS)")
    end_time: str = Field(description="End time of the part (HH:MM:SS)")
    phrases: list[str] = Field(
        description="Sample utterances from the part to orient the user"
    )
    audio_hash: str | None = None


class Utterance(BaseModel):
    """A single utterance captured in a tape part."""

    speaker: str = Field(description="Speaker identifier (e.g. 'S1', 'Moderator')")
    text: str = Field(description="The transcribed text")
    audio_hash: str | None = None


class Part(BaseModel):
    """A contiguous section of the tape containing utterances."""

    start_time: str = Field(description="Start time of the part (HH:MM:SS)")
    end_time: str = Field(description="End time of the part (HH:MM:SS)")
    audio_hash: str | None = None
    utterances: list[Utterance] = Field(default_factory=list)


class Tape(BaseModel):
    """A tape with its metadata and transcribed parts."""

    id: str
    filename: str
    audio_hash: str | None = None
    duration: str = Field(
        default="00:00:00",
        description="Total duration of the tape (HH:MM:SS)",
    )
    current_position: str = Field(
        default="00:00:00",
        description="Current position while processing (HH:MM:SS)",
    )
    context_parts: int = Field(
        default=1,
        description="Number of earlier parts to provide as context when transcribing",
    )
    parts: list[Part] = Field(default_factory=list)


def get_data_dir() -> Path:
    """Return the path to the data directory."""

    return Path(os.getenv("IEVA_DATA", "/data"))


def get_tape(tape_id: str) -> Tape:
    """Load a tape by ID."""

    return find(Tape, tape_id)


def save_tape(tape: Tape) -> str:
    """Persist a tape and return its identifier."""

    return save(tape)


def list_tapes() -> list[Tape]:
    """Return every stored tape."""

    return store_list_models(Tape)

