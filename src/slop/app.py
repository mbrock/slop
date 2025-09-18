import logging
from functools import wraps
from typing import TypedDict

import tagflow
from httpx import AsyncClient
from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from tagflow import TagResponse

import slop.gemini as gemini
import slop.store as store
from slop import transcribe

logger = logging.getLogger("slop.app")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class AppParams(TypedDict):
    client: AsyncClient
    google_api_key: str
    gemini_model: str
    interviews_db_path: str
    blobs_db_path: str


def html(func):
    """Ensure TagFlow handlers always return a response."""

    @wraps(func)
    async def wrapper(request: Request) -> Response:
        result = await func(request)
        if result is None:
            return TagResponse()
        if isinstance(result, Response):
            return result
        raise TypeError(
            f"Handler {func.__name__} returned unsupported type {type(result)!r}"
        )

    return wrapper


class AppParamsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, params: AppParams | None = None):
        super().__init__(app)
        self.state = params

    async def dispatch(self, request, call_next):
        state: AppParams = self.state

        with (
            gemini.api_key.using(state["google_api_key"]),
            gemini.http_client.using(state["client"]),
            gemini.model.using(state["gemini_model"]),
            store.with_databases(state["interviews_db_path"], state["blobs_db_path"]),
        ):
            return await call_next(request)


@html
async def get_home(_: Request):
    transcribe.home()


@html
async def get_home_partial(_: Request):
    transcribe.render_home_content()


@html
async def list_interviews(_: Request):
    transcribe.render_interview_list()


@html
async def upload_endpoint(request: Request):
    async with request.form() as form:
        audio = form.get("audio")
        if not isinstance(audio, UploadFile):
            raise HTTPException(status_code=400, detail="Audio file missing")
        await transcribe.upload_audio(audio)


@html
async def get_interview(request: Request):
    interview_id = request.path_params["id"]
    transcribe.view_interview(interview_id)


@html
async def view_segment_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    transcribe.view_segment(interview_id, segment_index)


@html
async def edit_segment_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    transcribe.edit_segment_dialog(interview_id, segment_index)


async def get_audio(request: Request):
    hash_ = request.path_params["hash_"]
    range_header = request.headers.get("range")
    transcribe.get_audio(hash_, range_header)


@html
async def transcribe_next_endpoint(request: Request):
    interview_id = request.path_params["id"]
    await transcribe.transcribe_next_segment(interview_id)


@html
async def retranscribe_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    await transcribe.retranscribe_segment(interview_id, segment_index)


@html
async def improve_speakers_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    form = await request.form()
    hint = form.get("hint")
    if hint is not None and not isinstance(hint, str):
        raise HTTPException(status_code=400, detail="Invalid hint")
    await transcribe.improve_speaker_identification(
        interview_id,
        segment_index,
        hint if isinstance(hint, str) else None,
    )


@html
async def update_segment_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    form = await request.form()
    content = form.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="Missing content")
    await transcribe.update_segment(interview_id, segment_index, content)


async def rename_interview_endpoint(request: Request):
    interview_id = request.path_params["id"]
    form = await request.form()
    new_name = form.get("new_name")
    if not isinstance(new_name, str):
        raise HTTPException(status_code=400, detail="New name required")
    transcribe.rename_interview(interview_id, new_name)


async def export_interview_endpoint(request: Request) -> Response:
    interview_id = request.path_params["id"]
    return transcribe.export_interview(interview_id)


def form_int(form: FormData, key: str) -> int:
    value = form.get(key)
    if value is None:
        raise HTTPException(status_code=400, detail=f"Missing form field {key}")
    try:
        return int(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid integer for {key}")


def form_str(form: FormData, key: str) -> str:
    value = form.get(key)
    if not isinstance(value, str):
        raise HTTPException(
            status_code=400, detail=f"Missing or invalid form field {key}"
        )
    return value


async def update_context_segments_endpoint(request: Request):
    interview_id = request.path_params["id"]
    async with request.form() as form:
        context_segments = form_int(form, "context_segments")
    transcribe.update_context_segments(interview_id, context_segments)


@html
async def update_speaker_endpoint(request: Request):
    interview_id = request.path_params["id"]
    segment_index = request.path_params["idx"]
    async with request.form() as form:
        utterance_index = form_int(form, "utterance_index")
        key = form_str(form, "key")

    transcribe.update_speaker(interview_id, segment_index, utterance_index, key)


def GET(path: str, endpoint):
    return Route(path, endpoint, methods=["GET"])


def POST(path: str, endpoint):
    return Route(path, endpoint, methods=["POST"])


def PUT(path: str, endpoint):
    return Route(path, endpoint, methods=["PUT"])


routes = [
    GET("/", get_home),
    GET("/home", get_home_partial),
    GET("/interview-list", list_interviews),
    GET("/audio/{hash_}", get_audio),
    POST("/upload", upload_endpoint),
    Mount(
        "/interview",
        routes=[
            GET("/{id}", get_interview),
            GET("/{id}/export", export_interview_endpoint),
            POST("/{id}/transcribe-next", transcribe_next_endpoint),
            POST("/{id}/rename", rename_interview_endpoint),
            POST("/{id}/context-segments", update_context_segments_endpoint),
            Mount(
                "/segment",
                routes=[
                    GET("/{idx:int}", view_segment_endpoint),
                    GET("/{idx:int}/edit", edit_segment_endpoint),
                    POST("/{idx:int}/retranscribe", retranscribe_endpoint),
                    POST("/{idx:int}/improve-speakers", improve_speakers_endpoint),
                    PUT("/{idx:int}", update_segment_endpoint),
                    POST("/{idx:int}/update-speaker", update_speaker_endpoint),
                ],
            ),
        ],
    ),
]


def create_app(app_state: AppParams) -> Starlette:
    application = Starlette(routes=routes)
    application.add_middleware(AppParamsMiddleware, app_state=app_state)
    application.add_middleware(tagflow.DocumentMiddleware)
    return application
