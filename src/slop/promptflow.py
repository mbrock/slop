"""Utilities for composing Gemini message content with a TagFlow-inspired API."""

from __future__ import annotations

from contextlib import contextmanager
from html import escape as html_escape
from typing import (
    AsyncIterator,
    Iterable,
    Mapping,
)

from slop import gemini
from slop.gemini import (
    Content,
    File,
    FileData,
    GenerateContentResponse,
    Part,
)
from slop.parameter import Parameter


def _escape_text(value: str) -> str:
    """Escape XML-special characters inside element text."""
    return html_escape(value, quote=False)


def _escape_attr(value: object) -> str:
    """Escape XML-special characters inside attribute values."""
    return html_escape(str(value), quote=True)


class FormattedContentBuffer:
    """A buffer that accumulates formatted text and converts it to Gemini parts.

    This maintains indentation state and handles the complexity of merging
    adjacent text content into parts.
    """

    def __init__(self, *, role: str | None = None) -> None:
        self.role = role
        self.indent_level = 0
        self.parts: list[Part] = []
        self.buffer: list[str] = []

    def write(self, text: str) -> None:
        """Write raw text to the buffer."""
        self.buffer.append(text)

    def write_line(self, text: str = "") -> None:
        """Write a line with current indentation."""
        if auto_format.get() and self._needs_newline():
            self.buffer.append("\n")
        if auto_format.get() and self.indent_level:
            self.buffer.append(indent_str.get() * self.indent_level)
        self.buffer.append(text)
        self.buffer.append("\n")

    def indent(self) -> None:
        """Increase indentation level."""
        self.indent_level += 1

    def dedent(self) -> None:
        """Decrease indentation level."""
        self.indent_level = max(0, self.indent_level - 1)

    def flush(self) -> None:
        """Flush buffer content to parts, merging with last text part if possible."""
        if not self.buffer:
            return
        text = "".join(self.buffer)
        self.buffer.clear()
        if not text:
            return

        # Merge with last part if it's text
        if self.parts and self.parts[-1].text is not None:
            existing = self.parts[-1].text or ""
            self.parts[-1].text = existing + text
        else:
            self.parts.append(Part(text=text))

    def add_part(self, part: Part) -> None:
        """Add a non-text part, flushing buffer first."""
        self.flush()
        self.parts.append(part)

    def get_parts(self) -> list[Part]:
        """Get all parts, flushing any remaining buffer."""
        self.flush()
        return list(self.parts)

    def to_content(self, role: str | None = None) -> Content:
        """Build a Content object with the specified or default role."""
        final_role = role or self.role
        if not final_role:
            raise ValueError("A role must be provided to build Content")
        return Content(role=final_role, parts=self.get_parts())

    def _needs_newline(self) -> bool:
        """Check if we need a newline before the next line."""
        last_char = self._last_character()
        return bool(last_char and last_char != "\n")

    def _last_character(self) -> str | None:
        """Get the last character in buffer or parts."""
        # Check buffer first
        for chunk in reversed(self.buffer):
            if chunk:
                return chunk[-1]
        # Then check parts
        for part in reversed(self.parts):
            if part.text and part.text:
                return part.text[-1]
        return None


# Parameters for context management with sensible defaults
_current_builder = Parameter[FormattedContentBuffer]("promptflow_current_builder")
current_contents = Parameter[list[Content]]("promptflow_current_contents")
auto_format = Parameter[bool]("promptflow_auto_format")
indent_str = Parameter[str]("promptflow_indent_str")

# Set defaults
auto_format._var.set(True)
indent_str._var.set("  ")


def _require_builder() -> FormattedContentBuffer:
    """Get the current builder or raise if none is active."""
    builder = _current_builder.peek()
    if builder is None:
        raise RuntimeError("No active FormattedContentBuffer context")
    return builder


def _require_contents() -> list[Content]:
    """Get the current contents list or raise if none is active."""
    contents = current_contents.peek()
    if contents is None:
        raise RuntimeError("No active contents list")
    return contents


# Nice public API functions that compose the core functionality


def text(value: str | None, *, escape: bool = True, indent: bool = False) -> None:
    """Append text to the current builder."""
    if not value:
        return
    builder = _require_builder()
    text_value = _escape_text(value) if escape else value
    if auto_format.get() and indent:
        if builder._needs_newline():
            builder.write("\n")
        if builder.indent_level:
            builder.write(indent_str.get() * builder.indent_level)
    builder.write(text_value)


def raw(value: str | None) -> None:
    """Append raw (unescaped) text."""
    text(value, escape=False)


def line(value: str | None = "", *, escape: bool = True) -> None:
    """Append a line of text."""
    if value is None:
        value = ""
    builder = _require_builder()
    text_value = _escape_text(value) if escape else value
    builder.write_line(text_value)


def blank_line() -> None:
    """Insert an empty line."""
    line("")


@contextmanager
def tag(
    name: str,
    attrs: Mapping[str, object] | None = None,
    **extra_attrs: object,
):
    """Context manager for XML tags."""
    builder = _require_builder()
    attributes = dict(attrs or {})
    attributes.update(extra_attrs)
    attr_fragment = "".join(
        f' {key}="{_escape_attr(value)}"' for key, value in attributes.items()
    )

    builder.write_line(f"<{name}{attr_fragment}>")
    builder.indent()
    try:
        yield builder
    finally:
        builder.dedent()
        builder.write_line(f"</{name}>")


def file(file: File | FileData | str, *, mime_type: str | None = None) -> None:
    """Insert a file reference as a part."""
    builder = _require_builder()

    # Build the FileData
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
    else:
        raise TypeError("file must be a File, FileData, or file URI string")

    # Add some formatting if needed
    if auto_format.get():
        if builder._needs_newline():
            builder.write("\n")
        if builder.indent_level:
            builder.write(indent_str.get() * builder.indent_level)

    # Add the file part
    builder.add_part(Part(fileData=file_data))

    # Ensure next content starts on new line
    if auto_format.get():
        builder.write("\n")


def extend(parts: Iterable[Part]) -> None:
    """Extend the builder with prebuilt parts."""
    builder = _require_builder()
    builder.flush()
    for part in parts:
        if part.text:
            # Text parts get merged properly
            builder.write(part.text)
            builder.flush()
        else:
            builder.add_part(part)


def append_part(part: Part) -> None:
    """Append an already constructed part."""
    extend([part])


@contextmanager
def message(*, role: str = "user"):
    """Context manager for building a message."""
    builder = FormattedContentBuffer(role=role)
    with _current_builder.using(builder):
        yield builder
        builder.flush()


@contextmanager
def parts_builder():
    """Context manager for building parts without a role."""
    builder = FormattedContentBuffer(role=None)
    with _current_builder.using(builder):
        yield builder
        builder.flush()


@contextmanager
def new_chat():
    """Context manager that creates a new conversation."""
    contents: list[Content] = []
    with current_contents.using(contents):
        yield contents


@contextmanager
def turn(role: str):
    """Add a conversation turn with the specified role."""
    contents = _require_contents()
    with message(role=role) as builder:
        yield builder
        contents.append(builder.to_content())


@contextmanager
def from_user():
    """Add a user turn to the conversation."""
    with turn("user") as builder:
        yield builder


@contextmanager
def from_model():
    """Add a model turn to the conversation."""
    with turn("model") as builder:
        yield builder


def append_content(content: Content) -> None:
    """Append a content to the current conversation."""
    contents = _require_contents()
    contents.append(content)


def extend_contents(contents_to_add: Iterable[Content]) -> None:
    """Extend the current conversation with multiple contents."""
    contents = _require_contents()
    contents.extend(contents_to_add)


async def generate() -> GenerateContentResponse:
    """Generate content using the current conversation."""
    contents = _require_contents()
    response = await gemini.generate(contents)

    if response.candidates:
        contents.append(response.candidates[0].content.model_copy(deep=True))

    return response


async def generate_streaming() -> AsyncIterator[GenerateContentResponse]:
    """Generate streaming content using the current conversation."""
    contents = _require_contents()

    with from_model():
        async for response in gemini.generate_streaming(contents):
            if response.candidates:
                extend(response.candidates[0].content.parts or [])
            yield response
