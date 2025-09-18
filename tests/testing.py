import functools
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import anyio.abc

from slop.parameter import Parameter


def test[**P, R](fn: Callable[P, R]):
    name = getattr(fn, "__name__", repr(fn))

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"❐ {name}")
        x = fn(*args, **kwargs)
        if hasattr(x, "__await__"):
            x = await x  # type: ignore
        print(f"✔︎ {name}")
        return x

    return wrapper


current_task_group = Parameter[anyio.abc.TaskGroup]("current_task_group")


@asynccontextmanager
async def scope():
    async with anyio.create_task_group() as tg:
        with current_task_group.using(tg):
            yield tg


def spawn(task: Callable[[], Awaitable[None]]):
    tg = current_task_group.get()
    if tg is None:
        raise RuntimeError("spawn() called outside of a scope() context")
    tg.start_soon(task)
