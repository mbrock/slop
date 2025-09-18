# AGENTS GUIDE

## Repository Purpose
- Web application for managing and transcribing ethnographic interview audio.
- Provides upload, segmentation, transcription, and speaker-identification workflows powered by Google Gemini models.
- Ships as a FastAPI app rendered with TagFlow/HTMX, backed by lightweight SQLite blob stores.

## High-Level Architecture
- **FastAPI + TagFlow UI (`src/slop/transcribe.py`)** renders HTML through Python context managers and serves HTMX interactions for uploads, segment editing, and progress updates.
- **Gemini integration (`src/slop/gemini.py`)** wraps the Google Generative Language API for audio uploads, content generation, and streaming responses; includes resumable uploads and error handling.
- **Transcription orchestrators (`src/slop/sndprompt.py`)** handle audio slicing with ffmpeg, prompt construction for Gemini, XML parsing of transcripts, and optional speaker-relabeling passes.
- **Domain models (`src/slop/models.py`)** defines Pydantic models for interviews, segments, and utterances.
- **Storage layer (`src/slop/store.py`)** provides generic blob/KV store functionality with SQLite-backed implementations: a JSON key-value `KeyValueStore` and binary `BlobStore`.
- **Presentation helpers (`src/slop/views.py`)** centralize reusable UI fragments such as upload areas, audio players, and styling rules.

## Checking, Linting, Formatting, Fixing

Type check with `uvx ty check`, lint with `uvx ruff check`, fix with `uvx ruff check --fix`,
format with `uvx ruff format`.

## Testing

Run the test suite with `python -m tests.run_all`. 

The test system uses a custom test runner (`tests/run_all.py`) that executes test modules and prints results with checkmarks. Test modules include:
- **`tests/gemini_tests.py`** - Tests for Gemini API integration including text generation, streaming, function calling, thinking mode, file uploads, and multi-turn conversations
- **`tests/test_transcribe.py`** - Tests for the FastAPI transcribe endpoints including home, interview list, and error handling for nonexistent resources
- **`tests/test_promptflow_lazy_files.py`** - Tests for the promptflow lazy file handling and blob store integration

Tests use a shared `tests/testing.py` module that provides utilities for creating test conversations and managing test state.

## Data & Persistence
- Audio and transcript metadata live under the `IEVA_DATA` directory (defaults to `/data`).
- `BLOBS` table stores content-addressed binary payloads (e.g., processed audio segments) hashed with SHA-256.
- `INTERVIEWS` table tracks interview progress, model selection, and segmented transcripts as serialized Pydantic documents.
- `backup.sh` snapshots `interviews.db` into `~/backups/interviews_db` via SQLite's `.backup` command.

## External Dependencies & Services
- Requires `ffmpeg` for audio normalization and segment extraction; installed in the provided Dockerfile image.
- Depends on `GOOGLE_API_KEY` for Gemini access; uploads use resumable sessions and deduplicate by hash.
- Async stack built on AnyIO, HTTPX (with `httpx-sse`), TagFlow, and Pydantic v2.
- UI assets leverage CDN-hosted Tailwind, HTMX, HyperScript, and Media Chrome components at runtime.

## Runtime Workflow
1. User uploads audio through the TagFlow form; `/upload` normalizes it to Ogg Vorbis, stores the blob, and registers an `Interview` record.
2. Viewing `/interview/{id}` streams segment cards, progress bars, and playback controls.
3. Transcription requests (`/interview/{id}/transcribe-next` and related endpoints) slice audio via `extract_segment`, send contextual prompts to Gemini, and persist returned utterances.
4. Segments support manual editing, re-transcription, and speaker re-labeling (`improve_speaker_identification_segment`).
5. Audio bytes are served back through `/audio/{hash}` with range support for in-browser playback.

## Development & Ops Notes
- Python 3.13 project managed with `uv`; lockfile `uv.lock` ensures deterministic installs.
- Docker image bundles dependencies, sets `IEVA_DATA=/data`, and launches `uv run uvicorn slop.app:app --host 0.0.0.0 --port 8080`.
- Local development works well with `uv run uvicorn slop.app:app --reload`.
- `tagflow-demo.py` offers a lightweight example of TagFlow usage outside the main app.
