"""Convenient context managers for configuring Gemini generation."""

from contextlib import contextmanager
from typing import Any, Literal

from slop import gemini
from slop.gemini import (
    FunctionCallingConfig,
    GenerationConfig,
    ThinkingConfig,
    Tool,
    ToolConfig,
)


def _merge_config(**kwargs: Any) -> GenerationConfig:
    """Merge new config values with existing generation config."""
    current = gemini.generation_config.peek()
    if current:
        # Create a copy and update with new values
        config_dict = dict(current)
        config_dict.update(kwargs)
        return config_dict  # type: ignore
    else:
        return kwargs  # type: ignore


@contextmanager
def temperature(value: float):
    """Set temperature for generation."""
    config = _merge_config(temperature=value)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def max_output(tokens: int):
    """Set max output tokens."""
    config = _merge_config(maxOutputTokens=tokens)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def top_p(value: float):
    """Set top_p for generation."""
    config = _merge_config(topP=value)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def top_k(value: int):
    """Set top_k for generation."""
    config = _merge_config(topK=value)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def stop_sequences(*sequences: str):
    """Set stop sequences."""
    config = _merge_config(stopSequences=list(sequences))
    with gemini.generation_config.using(config):
        yield


@contextmanager
def response_mime_type(mime: str):
    """Set response MIME type (e.g., 'application/json')."""
    config = _merge_config(responseMimeType=mime)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def thinking(include_thoughts: bool = True, budget: int | None = None):
    """Configure thinking mode.

    Args:
        include_thoughts: Whether to include thought summaries in response
        budget: Thought token budget (0=disabled, -1=dynamic, >0=fixed)
    """
    thinking_config: ThinkingConfig = {
        "includeThoughts": include_thoughts,
        "thinkingBudget": budget,
    }
    config = _merge_config(thinkingConfig=thinking_config)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def tools(*tool_list: Tool):
    """Set tools for generation."""
    with gemini.tools.using(list(tool_list)):
        yield


@contextmanager
def tool_config(
    mode: Literal["ANY", "AUTO", "NONE"] = "AUTO",
    allowed_functions: list[str] | None = None,
):
    """Configure tool calling behavior.

    Args:
        mode: "ANY", "AUTO", or "NONE"
        allowed_functions: Optional list of allowed function names
    """
    func_config: FunctionCallingConfig = {
        "mode": mode,
        "allowedFunctionNames": allowed_functions,
    }
    config: ToolConfig = {"functionCallingConfig": func_config}
    with gemini.tool_config.using(config):
        yield


@contextmanager
def model(name: str):
    """Set the model to use."""
    with gemini.model.using(name):
        yield


@contextmanager
def api_key(key: str):
    """Set the API key."""
    with gemini.api_key.using(key):
        yield


@contextmanager
def generation(**kwargs: Any):
    """Set arbitrary generation config values."""
    config = _merge_config(**kwargs)
    with gemini.generation_config.using(config):
        yield


@contextmanager
def json_mode():
    """Convenience for JSON response mode."""
    with response_mime_type("application/json"):
        yield
