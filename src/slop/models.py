import hashlib
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from slop.parameter import Parameter


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


# Parameters for database connections
interviews_db: Parameter[sqlite3.Connection] = Parameter("interviews_db")
blobs_db: Parameter[sqlite3.Connection] = Parameter("blobs_db")


def get_data_dir() -> Path:
    """Get the data directory from environment or default."""
    return Path(os.getenv("IEVA_DATA", "/data"))


@contextmanager
def sqlite_connection(db_path: str):
    """Context manager for SQLite database connection."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_databases():
    interviews_db.get().execute(
        """
        CREATE TABLE IF NOT EXISTS store (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    blobs_db.get().execute(
        """
        CREATE TABLE IF NOT EXISTS blobs (
            hash TEXT PRIMARY KEY,
            data BLOB NOT NULL,
            mime_type TEXT NOT NULL
        )
        """
    )


# Interview functions
def get_interview(interview_id: str) -> Interview | None:
    """Get an interview by ID."""
    conn = interviews_db.get()
    cursor = conn.execute("SELECT value FROM store WHERE key = ?", (interview_id,))
    if row := cursor.fetchone():
        return Interview.model_validate_json(row[0])
    return None


def save_interview(interview: Interview) -> None:
    """Save an interview."""
    conn = interviews_db.get()
    conn.execute(
        "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
        (interview.id, interview.model_dump_json()),
    )
    conn.commit()


def list_interviews() -> list[Interview]:
    """Get all interviews."""
    conn = interviews_db.get()
    cursor = conn.execute("SELECT value FROM store")
    return [Interview.model_validate_json(row[0]) for row in cursor.fetchall()]


# Blob functions
def save_blob(data: bytes, mime_type: str) -> str:
    """Store binary data and return its hash."""
    hash_ = hashlib.sha256(data).hexdigest()
    conn = blobs_db.get()
    conn.execute(
        "INSERT OR IGNORE INTO blobs (hash, data, mime_type) VALUES (?, ?, ?)",
        (hash_, data, mime_type),
    )
    conn.commit()
    return hash_


def get_blob(hash_: str) -> tuple[bytes, str] | None:
    """Get binary data and mime type by hash."""
    conn = blobs_db.get()
    cursor = conn.execute("SELECT data, mime_type FROM blobs WHERE hash = ?", (hash_,))
    if row := cursor.fetchone():
        return row[0], row[1]
    return None
