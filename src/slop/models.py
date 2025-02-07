import hashlib
import os
from pathlib import Path
import sqlite3
from pydantic import BaseModel, Field


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


class BlobStore:
    """
    Content-addressed store for binary data.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database with a blobs table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    mime_type TEXT NOT NULL
                )
                """
            )

    def put(self, data: bytes, mime_type: str) -> str:
        """Store binary data and return its hash."""
        hash_ = hashlib.sha256(data).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blobs (hash, data, mime_type) VALUES (?, ?, ?)",
                (hash_, data, mime_type),
            )
        return hash_

    def get(self, hash_: str) -> tuple[bytes, str] | None:
        """Get binary data and mime type by hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT data, mime_type FROM blobs WHERE hash = ?", (hash_,)
            )
            if row := cursor.fetchone():
                return row[0], row[1]
            return None


data_dir = os.getenv("IEVA_DATA", "/data")


class Store:
    """
    A simple key-value store using SQLite and a Pydantic model.
    """

    def __init__(self, db_path: str | Path, model_class: type[BaseModel]):
        self.db_path = Path(db_path)
        self.model_class = model_class
        self._init_db()

    def _init_db(self):
        """Initialize the database with a key-value table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> BaseModel | None:
        """Get a value by key."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store WHERE key = ?", (key,))
            if row := cursor.fetchone():
                return self.model_class.model_validate_json(row[0])
            return None

    def put(self, key: str, value: BaseModel):
        """Store a value by key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
                (key, value.model_dump_json()),
            )

    def values(self) -> list[BaseModel]:
        """Get all values."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store")
            return [
                self.model_class.model_validate_json(row[0])
                for row in cursor.fetchall()
            ]


INTERVIEWS = Store(Path(data_dir, "interviews.db"), Interview)
BLOBS = BlobStore(Path(data_dir, "blobs.db"))
