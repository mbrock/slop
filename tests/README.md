# Gemini API Tests

This directory contains comprehensive tests for the Gemini API client using pytest.

## Requirements

- Set `GOOGLE_API_KEY` environment variable
- Python 3.11+
- pytest and pytest-asyncio (installed via `uv add --dev pytest pytest-asyncio`)

## Test Structure

```
tests/
├── conftest.py                     # Pytest configuration
└── gemini/
    ├── test_basic_generation.py    # Basic text generation
    ├── test_streaming.py           # Streaming responses
    ├── test_conversation.py        # Multi-turn conversations
    ├── test_file_upload.py         # File upload and processing
    ├── test_function_calling.py    # Function calling
    ├── test_error_handling.py      # Error scenarios
    └── test_list_files.py          # File listing
```

## Running Tests

### Run All Tests
```bash
pytest

# With verbose output
pytest -v

# With captured output shown
pytest -s
```

### Run Specific Test Module
```bash
pytest tests/gemini/test_basic_generation.py
pytest tests/gemini/test_streaming.py
```

### Run Specific Test
```bash
pytest tests/gemini/test_basic_generation.py::test_basic_text_generation
```

### Run with Markers
```bash
# Run only async tests (all our tests)
pytest -m asyncio
```

## Notes

- Tests use the Gemini 2.5 Flash model by default
- File upload tests create temporary files that are automatically cleaned up
- Error handling tests intentionally trigger errors to verify proper handling
- Some tests may fail due to rate limits or model availability