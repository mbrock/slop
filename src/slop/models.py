"""Domain models for the tape transcription workflow."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from pydantic import BaseModel, Field

from slop.store import (
    ModelDecodeError,
    ModelNotFoundError,
    database,
    find,
    list_models as store_list_models,
    save,
)


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

    id: str
    tape_id: str
    start_time: str = Field(description="Start time of the part (HH:MM:SS)")
    end_time: str = Field(description="End time of the part (HH:MM:SS)")
    audio_hash: str | None = None
    utterances: list[Utterance] = Field(default_factory=list)


class Tape(BaseModel):
    """A tape with its metadata and transcribed part identifiers."""

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
    part_ids: list[str] = Field(
        default_factory=list,
        description="Ordered identifiers of parts belonging to the tape",
    )


_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_DEFAULT_ID_LENGTH = 10

_TAPE_KEY_PREFIX = "tape:"
_PART_KEY_PREFIX = "part:"


def generate_id(length: int = _DEFAULT_ID_LENGTH) -> str:
    """Return a random uppercase base32 identifier."""

    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(length))


def _tape_key(tape_id: str) -> str:
    return f"{_TAPE_KEY_PREFIX}{tape_id}"


def _part_key(part_id: str) -> str:
    return f"{_PART_KEY_PREFIX}{part_id}"


def get_data_dir() -> Path:
    """Return the path to the data directory."""

    return Path(os.getenv("IEVA_DATA", "/data"))


def get_tape(tape_id: str) -> Tape:
    """Load a tape by ID."""

    try:
        return find(Tape, _tape_key(tape_id))
    except ModelNotFoundError:
        return _load_legacy_tape(tape_id)


def save_tape(tape: Tape) -> str:
    """Persist a tape and return its identifier."""

    return save(tape, key=_tape_key(tape.id))


def list_tapes() -> list[Tape]:
    """Return every stored tape."""

    tapes = store_list_models(Tape, key_prefix=_TAPE_KEY_PREFIX)
    if tapes:
        return tapes

    migrated = False
    for key, raw in database.get().store.items():
        if key.startswith(_TAPE_KEY_PREFIX) or key.startswith(_PART_KEY_PREFIX):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "filename" not in payload:
            continue
        if "parts" not in payload:
            continue
        try:
            _load_legacy_tape(key)
        except (ModelNotFoundError, ModelDecodeError):
            continue
        migrated = True

    return store_list_models(Tape, key_prefix=_TAPE_KEY_PREFIX) if migrated else tapes


def get_part(part_id: str) -> Part:
    """Load a part by ID."""

    return find(Part, _part_key(part_id))


def save_part(part: Part) -> str:
    """Persist a part and return its identifier."""

    return save(part, key=_part_key(part.id))


def list_parts(tape_id: str | None = None) -> list[Part]:
    """Return all stored parts, optionally filtered by tape."""

    parts = store_list_models(Part, key_prefix=_PART_KEY_PREFIX)
    if tape_id is None:
        return parts
    return [part for part in parts if part.tape_id == tape_id]


def list_parts_for_tape(tape: Tape) -> list[Part]:
    """Load parts for the given tape in stored order."""

    parts: list[Part] = []
    for part_id in tape.part_ids:
        parts.append(get_part(part_id))
    return parts


def new_tape_id() -> str:
    """Generate an unused tape identifier."""

    while True:
        candidate = generate_id()
        try:
            get_tape(candidate)
        except ModelNotFoundError:
            return candidate


def new_part_id() -> str:
    """Generate an unused part identifier."""

    while True:
        candidate = generate_id()
        try:
            get_part(candidate)
        except ModelNotFoundError:
            return candidate


def _load_legacy_tape(tape_id: str) -> Tape:
    """Load and migrate a tape stored in the legacy format."""

    raw = database.get().store.get(tape_id)
    if raw is None:
        raise ModelNotFoundError(f"Tape with key '{tape_id}' not found")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelDecodeError(
            f"Legacy Tape with key '{tape_id}' could not be decoded"
        ) from exc

    if not isinstance(payload, dict):
        raise ModelDecodeError(
            f"Legacy Tape with key '{tape_id}' has invalid structure"
        )

    parts_payload = payload.pop("parts", [])
    tape = Tape.model_validate(payload)

    migrated_ids: list[str] = []
    for part_payload in parts_payload or []:
        if not isinstance(part_payload, dict):
            continue
        legacy_payload = dict(part_payload)
        legacy_payload["tape_id"] = tape.id
        legacy_payload["id"] = new_part_id()
        part = Part.model_validate(legacy_payload)
        save_part(part)
        migrated_ids.append(part.id)

    tape.part_ids = migrated_ids
    save_tape(tape)
    database.get().store.set(tape_id, tape.model_dump_json())
    return tape
