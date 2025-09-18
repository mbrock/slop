"""Extremely small async test runner with no dynamic imports."""

import anyio

from tests import gemini_tests, test_promptflow_lazy_files, test_transcribe


async def main():
    await test_promptflow_lazy_files.main()
    await test_transcribe.main()
    await gemini_tests.main()


if __name__ == "__main__":
    anyio.run(main)
