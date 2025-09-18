import hashlib
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

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


class KeyValueStore(Protocol):
    def get(self, key: str) -> str | None:
        """Return the JSON string stored for ``key`` or ``None`` if absent."""

    def set(self, key: str, value: str) -> None:
        """Persist the JSON ``value`` at ``key``."""

    def values(self) -> Iterable[str]:
        """Return an iterable of all stored JSON strings."""


class BlobStore(Protocol):
    def get(self, hash_: str) -> tuple[bytes, str] | None:
        """Return ``(data, mime_type)`` for ``hash_`` or ``None`` when missing."""

    def put(self, hash_: str, data: bytes, mime_type: str) -> None:
        """Persist ``data`` under ``hash_`` while recording ``mime_type``."""


class Database(Protocol):
    store: KeyValueStore
    blobs: BlobStore

    def init(self) -> None:
        """Initialize persistent storage (e.g., create tables)."""


database: Parameter[Database] = Parameter("database")


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


class SQLiteKeyValueStore:
    """SQLite-backed mapping for JSON-serialized models."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get(self, key: str) -> str | None:
        cursor = self._conn.execute("SELECT value FROM store WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def values(self) -> Iterable[str]:
        cursor = self._conn.execute("SELECT value FROM store")
        return [row[0] for row in cursor.fetchall()]


class SQLiteBlobStore:
    """SQLite-backed blob repository."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get(self, hash_: str) -> tuple[bytes, str] | None:
        cursor = self._conn.execute(
            "SELECT data, mime_type FROM blobs WHERE hash = ?",
            (hash_,),
        )
        if row := cursor.fetchone():
            return row[0], row[1]
        return None

    def put(self, hash_: str, data: bytes, mime_type: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO blobs (hash, data, mime_type) VALUES (?, ?, ?)",
            (hash_, data, mime_type),
        )
        self._conn.commit()


class SQLiteDatabase:
    """Concrete :class:`Database` backed by SQLite connections."""

    def __init__(self, store_conn: sqlite3.Connection, blob_conn: sqlite3.Connection):
        self._store_conn = store_conn
        self._blob_conn = blob_conn
        self.store: KeyValueStore = SQLiteKeyValueStore(store_conn)
        self.blobs: BlobStore = SQLiteBlobStore(blob_conn)

    def init(self) -> None:
        self._store_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._store_conn.commit()

        self._blob_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                hash TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                mime_type TEXT NOT NULL
            )
            """
        )
        self._blob_conn.commit()


@contextmanager
def with_databases(interviews_db_path: str, blobs_db_path: str):
    """Context manager to set up database connections."""

    with sqlite_connection(interviews_db_path) as interviews_conn:
        with sqlite_connection(blobs_db_path) as blobs_conn:
            sqlite_db = SQLiteDatabase(interviews_conn, blobs_conn)
            with database.using(sqlite_db):
                yield


def init_databases() -> None:
    database.get().init()


T = TypeVar("T", bound=BaseModel)


class ModelNotFoundError(LookupError):
    """Raised when a model is missing from the backing store."""


class ModelDecodeError(ValueError):
    """Raised when deserializing a stored model fails."""


def find(model: type[T], key: str) -> T:
    """Return the stored model for ``key`` or raise if it is missing/invalid."""

    if (raw := database.get().store.get(key)) is None:
        raise ModelNotFoundError(f"{model.__name__} with key '{key}' not found")

    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise ModelDecodeError(
            f"Stored {model.__name__} with key '{key}' could not be decoded"
        ) from exc


def save(model_instance: T, *, key: str | None = None) -> str:
    """Persist ``model_instance`` using ``key`` or its ``id`` attribute."""

    resolved_key = key or getattr(model_instance, "id", None)
    if not isinstance(resolved_key, str):
        raise ValueError("A string key (or model.id) is required to save the model")

    database.get().store.set(resolved_key, model_instance.model_dump_json())
    return resolved_key


def list_models(model: type[T]) -> list[T]:
    """Return every stored model of the requested type."""

    result: list[T] = []
    for raw in database.get().store.values():
        try:
            result.append(model.model_validate_json(raw))
        except ValidationError as exc:
            raise ModelDecodeError(
                f"Stored {model.__name__} value could not be decoded"
            ) from exc
    return result


def get_interview(interview_id: str) -> Interview:
    """Compatibility wrapper for loading :class:`Interview` instances."""

    return find(Interview, interview_id)


def save_interview(interview: Interview) -> str:
    """Compatibility wrapper for storing :class:`Interview` instances."""

    return save(interview)


def list_interviews() -> list[Interview]:
    """Compatibility wrapper returning every stored interview."""

    return list_models(Interview)


def save_blob(data: bytes, mime_type: str) -> str:
    """Store binary data and return its hash."""

    hash_ = hashlib.sha256(data).hexdigest()
    database.get().blobs.put(hash_, data, mime_type)
    return hash_


def get_blob(hash_: str) -> tuple[bytes, str] | None:
    """Get binary data and mime type by hash."""

    return database.get().blobs.get(hash_)
