"""Test file listing with Gemini API."""

import pytest
from src.slop.gemini import GeminiClient


@pytest.mark.asyncio
async def test_list_files():
    """Test listing files."""
    client = GeminiClient()
    
    file_list = await client.list_files(page_size=5)
    
    # Should return a FileList object
    assert hasattr(file_list, 'files')
    assert isinstance(file_list.files, list)
    
    # Check file attributes if any files exist
    if file_list.files:
        file = file_list.files[0]
        assert hasattr(file, 'name')
        assert hasattr(file, 'uri')