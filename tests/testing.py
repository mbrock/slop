import functools
import os
import sys
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import anyio.abc

from slop.parameter import Parameter


test_filter = Parameter[str | None]("test_filter")
_fallback_filter = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEST_FILTER")


def test[**P, R](fn: Callable[P, R]):
    name = getattr(fn, "__name__", repr(fn))

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        active_filter = test_filter.peek()
        if active_filter is None:
            active_filter = _fallback_filter

        if active_filter and active_filter not in name:
            print(f"• skipping {name} (filter '{active_filter}')")
            return None  # type: ignore[return-value]

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
