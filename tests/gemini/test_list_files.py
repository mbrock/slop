"""Test file listing with Gemini API."""

from src.slop import gemini


async def test_list_files():
    """Test listing files."""
    file_list = await gemini.list_files(page_size=5)

    # Should return a FileList object
    assert hasattr(file_list, "files")
    assert isinstance(file_list.files, list)

    # Check file attributes if any files exist
    if file_list.files:
        file = file_list.files[0]
        assert hasattr(file, "name")
        assert hasattr(file, "uri")
