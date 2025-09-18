"""Extremely small async test runner with no dynamic imports."""

import os
import sys
from contextlib import nullcontext

import anyio

from tests import gemini_tests, test_promptflow_lazy_files, test_transcribe
from tests.testing import scope, spawn, test_filter


@scope()
async def main():
    filter_value = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEST_FILTER")
    filter_context = test_filter.using(filter_value) if filter_value else nullcontext()

    with filter_context:
        spawn(test_promptflow_lazy_files.main)
        spawn(test_transcribe.main)
        spawn(gemini_tests.main)


if __name__ == "__main__":
    anyio.run(main)
