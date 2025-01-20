from contextlib import asynccontextmanager, contextmanager
import json
import logging
import sqlite3
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import rich
from tagflow import (
    DocumentMiddleware,
    Live,
    TagResponse,
    tag,
    text,
    attr,
)

from slop.gemini import (
    Content,
    FileData,
    FunctionCallingConfig,
    FunctionDeclaration,
    GenerateRequest,
    GeminiClient,
    Part,
    Tool,
    ToolConfig,
)
from slop.views import upload_area

logger = logging.getLogger("slop.transcribe")

# Initialize Live instance for WebSocket support
live = Live()


class PartitionSegment(BaseModel):
    """A segment of the interview with start and end times"""

    start_time: str = Field(description="Start time of the segment (HH:MM:SS)")
    end_time: str = Field(description="End time of the segment (HH:MM:SS)")
    phrases: list[str] = Field(
        description="Some example short phrases from this segment to help orient the user"
    )


class Interview(BaseModel):
    """An interview with its metadata and segments"""

    id: str
    filename: str
    file_uri: str
    segments: list[PartitionSegment] = []


class Store:
    """A simple key-value store using SQLite and Pydantic"""

    def __init__(self, db_path: str | Path, model_class: type[BaseModel]):
        self.db_path = Path(db_path)
        self.model_class = model_class
        self._init_db()

    def _init_db(self):
        """Initialize the database with a key-value table"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def get(self, key: str) -> BaseModel | None:
        """Get a value by key"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store WHERE key = ?", (key,))
            if row := cursor.fetchone():
                return self.model_class.model_validate_json(row[0])
            return None

    def put(self, key: str, value: BaseModel):
        """Store a value by key"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store (key, value) VALUES (?, ?)",
                (key, value.model_dump_json()),
            )

    def values(self) -> list[BaseModel]:
        """Get all values"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM store")
            return [
                self.model_class.model_validate_json(row[0])
                for row in cursor.fetchall()
            ]


# Initialize the store
INTERVIEWS = Store(Path("data/interviews.db"), Interview)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with live.run(app):
        logger.info("live server started")
        yield
    logger.info("live server stopped")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=TagResponse,
    title="Slop Interview",
)
app.add_middleware(DocumentMiddleware)


@contextmanager
def layout(title: str):
    """Common layout wrapper for all pages"""
    with tag.html(lang="en"):
        with tag.head():
            with tag.title():
                text(f"{title} - Slop")
            with tag.script(src="https://cdn.tailwindcss.com"):
                pass
            with tag.script(src="https://unpkg.com/htmx.org@1.9.10"):
                pass
            with tag.script():
                text("""htmx.config = { "globalViewTransitions": true }""")
            # set tailwind serif font to equity ot
            with tag.script():
                text("""
                    tailwind.config = {
                      theme: {
                        fontFamily: {
                            serif: ["Equity OT", "Times New Roman", "serif"],
                        },
                      },
                    }
                """)
            live.script_tag()

        with tag.body(classes="bg-stone-300 min-h-screen font-serif p-4"):
            yield


@app.get("/")
async def home():
    with layout("Home"):
        with tag.div(classes="prose mx-auto"):
            upload_area()


@app.post("/upload")
async def upload_audio(audio: UploadFile):
    # Save the uploaded file temporarily
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(audio.filename).suffix
    ) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp.flush()

        # Upload to Gemini
        client = GeminiClient()
        file = await client.upload_file(tmp.name, display_name=audio.filename)

        # Create interview record
        interview_id = str(len(list(INTERVIEWS.values())) + 1)
        interview = Interview(
            id=interview_id,
            filename=audio.filename,
            file_uri=file.uri,
        )
        INTERVIEWS.put(interview_id, interview)

        # Clean up temp file
        Path(tmp.name).unlink()

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


@app.get("/interview/{interview_id}")
async def view_interview(interview_id: str):
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    with layout(f"Interview - {interview.filename}"):
        with tag.div(classes="prose mx-auto"):
            if not interview.segments:
                with tag.form(
                    action=f"/interview/{interview_id}/partition", method="post"
                ):
                    with tag.button(
                        type="submit",
                        classes="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600",
                    ):
                        text("Partition")
            else:
                for segment in interview.segments:
                    with tag.article(
                        classes="bg-white py-1 px-4 border-b-2 border-stone-500 mb-2"
                    ):
                        with tag.p(classes="text-gray-900"):
                            with tag.span(classes="block mr-2"):
                                text(" — ".join(segment.phrases))


PARTITION_TOOL = Tool(
    functionDeclarations=[
        FunctionDeclaration(
            name="partition_interview",
            description="Partition an audio interview into segments of around a minute each",
            parameters={
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {
                                    "type": "string",
                                    "description": "Start time of the segment (HH:MM:SS)",
                                },
                                "end_time": {
                                    "type": "string",
                                    "description": "End time of the segment (HH:MM:SS)",
                                },
                                "phrases": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A brief set of phrases or paraphrases that help identify the segment. It should fit in a line.",
                                    },
                                },
                            },
                            "required": ["start_time", "end_time", "phrases"],
                        },
                        "description": "The segments of the interview (around a minute each)",
                    }
                },
                "required": ["segments"],
            },
        )
    ]
)


@app.post("/interview/{interview_id}/partition")
async def partition_interview(interview_id: str):
    if not (interview := INTERVIEWS.get(interview_id)):
        raise HTTPException(status_code=404, detail="Interview not found")

    client = GeminiClient()
    request = GenerateRequest(
        contents=[
            Content(
                role="user",
                parts=[
                    Part(fileData=FileData(fileUri=interview.file_uri)),
                    Part(
                        text="""Please listen to this interview and partition it into segments of around a minute each. 
                        Provide the start and end time for each segment, along with a brief set of phrases that help identify the segment.
                        Use the `partition_interview` tool to provide the segments."""
                    ),
                ],
            )
        ],
        tools=[PARTITION_TOOL],
        toolConfig=ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY")),
    )

    response = await client.generate_content_sync(request)
    if not response.candidates:
        raise HTTPException(status_code=500, detail="Failed to partition interview")

    rich.print(response)

    # Look for a function call part among all parts
    function_call = None
    for part in response.candidates[0].content.parts:
        rich.print(part)
        if hasattr(part, "functionCall") and part.functionCall:
            function_call = part.functionCall
            break

    if not function_call or function_call.name != "partition_interview":
        raise HTTPException(status_code=500, detail="Invalid response from model")

    # Update the interview with the new segments
    interview.segments = [
        PartitionSegment(**segment) for segment in function_call.args["segments"]
    ]
    INTERVIEWS.put(interview.id, interview)

    response = RedirectResponse(url=f"/interview/{interview_id}")
    response.status_code = 303  # See Other
    return response


if __name__ == "__main__":
    import trio
    import hypercorn.trio
    import hypercorn.config
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    config = hypercorn.config.Config()
    config.bind = ["localhost:8000"]
    config.worker_class = "trio"

    trio.run(hypercorn.trio.serve, app, config)
