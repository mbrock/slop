# Transcription Web App

I need to make a web app for transcribing long interviews.

Use FastAPI with Trio, Hypercorn, HTMX, Tailwind, and Tagflow.

## Workflow

Upload an audio file.

Ask Gemini to partition the interview into segments of a few speaker turns each.

Ask Gemini to transcribe a segment.

Edit the transcript, or ask Gemini to edit it.

Use previous segment audio/text pairs as context for the next segment.

## HTML, Streaming, etc

We ask Gemini to produce HTML, and we will want to parse the stream and turn it into Tagflow live DOM updates.

Example output:

```html
<span data-speaker="S1" data-time="00:00:08">How will the world end?</span>
<span data-speaker="S2" data-time="00:00:10"
  >Well, the sun will run out of fuel in about four billion years or so.</span
>
<span data-speaker="S2" data-time="00:00:16"
  >And actually before that, it will begin to— to swell up, expand, and so we
  think the Earth will get incinerated.</span
>
```

## Data Store

## Tagflow Overview

Tagflow is a Python library for generating HTML that aligns with Python's native control flow patterns. Key features:

- Uses context managers (`with` blocks) instead of nested function calls
- Allows natural use of Python control structures (if, for, try/except)
- Components can be created using decorators and context managers
- Includes FastAPI integration via custom response class and middleware
- Supports "live documents" with WebSocket-based DOM updates

Example component:

```python
@contextmanager
def transcript_segment():
    with tag.article(classes="prose mx-auto my-4"):
        yield

@html.span()
def speaker_turn(speaker: str, timestamp: str, text: str):
    attr("data-speaker", speaker)
    attr("data-time", timestamp)
    text(text)
```

The library's approach to HTML generation will work well with our streaming transcription needs, especially with its live document features for real-time updates.
