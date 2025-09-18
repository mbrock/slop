"""Utilities for composing Gemini message content with a TagFlow-inspired API."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from html import escape as html_escape
from typing import Awaitable, Callable, ContextManager, Iterable, Iterator, Mapping

import anyio

from slop.gemini import Content, File, FileData, Part
from slop.models import blobs_db, get_blob

# Mirrors TagFlow's global context tracking so helper functions can operate
# without passing the builder around explicitly.
_current_builder: ContextVar[GeminiMessageBuilder | None]
_current_builder = ContextVar("promptflow_current_builder")


def _escape_text(value: str) -> str:
    """Escape XML-special characters inside element text."""
    return html_escape(value, quote=False)


def _escape_attr(value: object) -> str:
    """Escape XML-special characters inside attribute values."""
    return html_escape(str(value), quote=True)


class GeminiMessageBuilder:
    """Helper for building Gemini message parts with XML-friendly utilities."""

    __slots__ = (
        "role",
        "_auto_format",
        "_indent",
        "_indent_level",
        "_parts",
        "_buffer",
        "_context_tokens",
    )

    def __init__(
        self,
        *,
        role: str | None,
        auto_format: bool = True,
        indent: str = "  ",
    ) -> None:
        self.role = role
        self._auto_format = auto_format
        self._indent = indent
        self._indent_level = 0
        self._parts: list[Part] = []
        self._buffer: list[str] = []
        self._context_tokens: list[Token[GeminiMessageBuilder | None]] = []

    # ------------------------------------------------------------------
    # context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "GeminiMessageBuilder":  # pragma: no cover - convenience
        token = _current_builder.set(self)
        self._context_tokens.append(token)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self._flush_text()
        token = self._context_tokens.pop()
        _current_builder.reset(token)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def text(
        self, value: str | None, *, escape: bool = True, indent: bool = False
    ) -> None:
        """Append raw text to the current buffer.

        Args:
            value: The text to append. `None` is ignored for convenience.
            escape: Escape XML characters (defaults to True).
            indent: If auto-formatting is enabled, indent this line to the current level.
        """
        if not value:
            return
        text_value = _escape_text(value) if escape else value
        if self._auto_format and indent:
            self._append_indent(self._indent_level)
        self._buffer.append(text_value)

    def raw(self, value: str | None) -> None:
        """Append raw (unescaped) text."""
        self.text(value, escape=False)

    def line(self, value: str | None = "", *, escape: bool = True) -> None:
        """Append a line of text, respecting indentation when auto-formatting."""
        if value is None:
            value = ""
        if self._auto_format:
            self._append_indent(self._indent_level)
        text_value = _escape_text(value) if escape else value
        self._buffer.append(text_value + "\n")

    def blank_line(self) -> None:
        """Insert an empty line respecting indentation."""
        if self._auto_format:
            self._append_indent(self._indent_level)
        self._buffer.append("\n")

    @contextmanager
    def tag(
        self,
        name: str,
        attrs: Mapping[str, object] | None = None,
        **extra_attrs: object,
    ):
        """Context manager that wraps content in opening and closing XML tags."""
        attributes = dict(attrs or {})
        attributes.update(extra_attrs)
        attr_fragment = "".join(
            f' {key}="{_escape_attr(value)}"' for key, value in attributes.items()
        )
        self._write_line(f"<{name}{attr_fragment}>")
        self._indent_level += 1
        try:
            yield self
        finally:
            self._indent_level = max(0, self._indent_level - 1)
            self._write_line(f"</{name}>")

    def file(
        self,
        file: File | FileData | str,
        *,
        mime_type: str | None = None,
    ) -> None:
        """Insert a file reference as its own part, keeping XML tags around it."""
        file_data: FileData
        if isinstance(file, FileData):
            file_data = (
                file.model_copy(update={"mimeType": mime_type})
                if mime_type
                else file.model_copy()
            )
        elif isinstance(file, File):
            file_data = FileData(
                fileUri=file.uri,
                mimeType=mime_type or file.mimeType,
            )
        elif isinstance(file, str):
            file_data = FileData(fileUri=file, mimeType=mime_type)
        else:  # pragma: no cover - defensive programming
            raise TypeError("file must be a File, FileData, or file URI string")

        if self._auto_format:
            self._append_indent(self._indent_level)
        self._flush_text()
        self._parts.append(Part(fileData=file_data))
        if self._auto_format:
            # ensure following text starts on a new line
            self._buffer.append("\n")

    def extend(self, parts: Iterable[Part]) -> None:
        """Extend the builder with prebuilt parts."""
        self._flush_text()
        for part in parts:
            if part.text:
                self._append_part_text(part.text)
            else:
                self._parts.append(part)

    def append_part(self, part: Part) -> None:
        """Append an already constructed part."""
        self.extend([part])

    def parts(self) -> list[Part]:
        """Return the assembled list of parts."""
        self._flush_text()
        return list(self._parts)

    def to_content(self, role: str | None = None) -> Content:
        """Convert the builder into a Gemini Content object."""
        final_role = role or self.role
        if not final_role:
            raise ValueError("A role must be provided to build Content")
        return Content(role=final_role, parts=self.parts())

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _write_line(self, line: str, *, indent_level: int | None = None) -> None:
        indent = self._indent_level if indent_level is None else indent_level
        if self._auto_format:
            self._append_indent(indent)
            self._buffer.append(line + "\n")
        else:
            self._buffer.append(line + "\n")

    def _append_indent(self, indent_level: int) -> None:
        if not self._auto_format:
            return
        if self._needs_newline():
            self._buffer.append("\n")
        if indent_level:
            self._buffer.append(self._indent * indent_level)

    def _needs_newline(self) -> bool:
        last_char = self._last_character()
        return bool(last_char and last_char != "\n")

    def _last_character(self) -> str | None:
        if self._buffer:
            for chunk in reversed(self._buffer):
                if chunk:
                    return chunk[-1]
        for part in reversed(self._parts):
            if part.text:
                text = part.text
                if text:
                    return text[-1]
        return None

    def _flush_text(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        if not text:
            return
        self._append_part_text(text)

    def _append_part_text(self, text: str) -> None:
        if self._parts and self._parts[-1].text is not None:
            existing = self._parts[-1].text or ""
            self._parts[-1].text = existing + text
        else:
            self._parts.append(Part(text=text))


def _require_builder() -> GeminiMessageBuilder:
    builder = _current_builder.get(None)
    if builder is None:
        raise RuntimeError("No active GeminiMessageBuilder context")
    return builder


@contextmanager
def tag(
    name: str,
    attrs: Mapping[str, object] | None = None,
    **extra_attrs: object,
):
    builder = _require_builder()
    with builder.tag(name, attrs, **extra_attrs) as ctx_builder:
        yield ctx_builder


def text(value: str | None, *, escape: bool = True, indent: bool = False) -> None:
    _require_builder().text(value, escape=escape, indent=indent)


def raw(value: str | None) -> None:
    _require_builder().raw(value)


def line(value: str | None = "", *, escape: bool = True) -> None:
    _require_builder().line(value, escape=escape)


def blank_line() -> None:
    _require_builder().blank_line()


def file(file: File | FileData | str, *, mime_type: str | None = None) -> None:
    _require_builder().file(file, mime_type=mime_type)


def extend(parts: Iterable[Part]) -> None:
    _require_builder().extend(parts)


def append_part(part: Part) -> None:
    _require_builder().append_part(part)


BlobUploader = Callable[[bytes, str], Awaitable[File]]


class ConversationBuilder:
    """Utility for collecting Gemini conversation turns via context managers."""

    __slots__ = ("_auto_format", "_indent", "_contents", "_upload")

    def __init__(
        self,
        *,
        auto_format: bool = True,
        indent: str = "  ",
        upload: BlobUploader | None,
    ) -> None:
        self._auto_format = auto_format
        self._indent = indent
        self._contents: list[Content] = []
        if upload is None:
            raise ValueError("upload callable is required")
        self._upload = upload

    @contextmanager
    def turn(self, role: str):
        builder = message(role=role, auto_format=self._auto_format, indent=self._indent)
        with builder:
            yield builder
        self._contents.append(builder.to_content())

    def user_turn(self) -> ContextManager[GeminiMessageBuilder]:
        return self.turn("user")

    def model_turn(self) -> ContextManager[GeminiMessageBuilder]:
        return self.turn("model")

    def append(self, content: Content) -> None:
        self._contents.append(content)

    def extend(self, contents: Iterable[Content]) -> None:
        self._contents.extend(contents)

    def to_contents(self) -> list[Content]:
        return list(self._contents)

    def __iter__(self) -> Iterator[Content]:
        return iter(self._contents)

    async def build_contents(self) -> list[Content]:
        """Resolve lazy blob references and return ready-to-send contents."""
        print("build_contents", blobs_db.peek())

        contents = [content.model_copy(deep=True) for content in self._contents]

        blobs: dict[str, list[Part]] = {}
        blob_mime_overrides: dict[str, str | None] = {}

        for content in contents:
            for part in content.parts:
                file_data = part.fileData
                if not file_data or not file_data.fileUri:
                    continue
                uri = file_data.fileUri
                if not uri.startswith("blob:"):
                    continue
                blob_hash = uri[len("blob:") :]
                if not blob_hash:
                    raise ValueError("Blob file URI must include a hash")
                blobs.setdefault(blob_hash, []).append(part)
                if file_data.mimeType:
                    existing = blob_mime_overrides.get(blob_hash)
                    if existing and existing != file_data.mimeType:
                        raise ValueError(
                            f"Conflicting MIME types for blob {blob_hash}:"
                            f" {existing!r} vs {file_data.mimeType!r}"
                        )
                    blob_mime_overrides[blob_hash] = file_data.mimeType

        if not blobs:
            return contents

        resolved_data: dict[str, tuple[str, str]] = {}

        async def resolve_blob(blob_hash: str) -> None:
            print(3, blobs_db.peek())
            blob = get_blob(blob_hash)
            if not blob:
                raise RuntimeError(f"Blob {blob_hash} not found in blob store")
            data, stored_mime = blob
            requested_mime = blob_mime_overrides.get(blob_hash) or stored_mime
            if not requested_mime:
                raise RuntimeError(f"Missing MIME type for blob {blob_hash}")
            uploaded = await self._upload(data, requested_mime)
            uri = uploaded.uri
            if not uri:
                raise RuntimeError(f"Upload for blob {blob_hash} returned empty URI")
            resolved_mime = (
                blob_mime_overrides.get(blob_hash)
                or uploaded.mimeType
                or requested_mime
            )
            if not resolved_mime:
                raise RuntimeError(f"Upload for blob {blob_hash} returned no MIME type")
            resolved_data[blob_hash] = (uri, resolved_mime)

        print(1, blobs_db.peek())
        async with anyio.create_task_group() as task_group:
            print(2, blobs_db.peek())
            for blob_hash in blobs:
                task_group.start_soon(resolve_blob, blob_hash)

        for blob_hash, parts in blobs.items():
            if blob_hash not in resolved_data:
                raise RuntimeError(f"No upload result for blob {blob_hash}")
            uri, resolved_mime = resolved_data[blob_hash]
            for part in parts:
                if not part.fileData:
                    continue
                requested_mime = part.fileData.mimeType or blob_mime_overrides.get(
                    blob_hash
                )
                part.fileData.fileUri = uri
                part.fileData.mimeType = requested_mime or resolved_mime

        return contents


def message(
    *,
    role: str = "user",
    auto_format: bool = True,
    indent: str = "  ",
) -> GeminiMessageBuilder:
    """Convenience factory for `with message(...) as prompt:` usage."""
    return GeminiMessageBuilder(role=role, auto_format=auto_format, indent=indent)


def parts_builder(
    *, auto_format: bool = True, indent: str = "  "
) -> GeminiMessageBuilder:
    """Create a builder when only individual parts (no role) are required."""
    return GeminiMessageBuilder(role=None, auto_format=auto_format, indent=indent)
