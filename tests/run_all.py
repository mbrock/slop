"""Extremely small async test runner with no dynamic imports."""

import anyio

from tests import gemini_tests, test_promptflow_lazy_files, test_transcribe
from tests.testing import scope, spawn


@scope()
async def main():
    spawn(test_promptflow_lazy_files.main)
    spawn(test_transcribe.main)
    spawn(gemini_tests.main)


if __name__ == "__main__":
    anyio.run(main)
