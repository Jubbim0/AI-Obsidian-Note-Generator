import pytest
import os
import sys
from pathlib import Path
import tempfile
import shutil
import logging
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gen_notes import (
    extract_text_from_pdf,
    extract_text_from_pptx,
    combine_inputs_to_text,
    create_obsidian_notes,
    setup_logging,
    main,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_pdf():
    """Get the sample PDF file."""
    return Path(__file__).parent / "sample_pdf" / "sample.pdf"


@pytest.fixture
def sample_pdf_text():
    """Get the expected text from the sample PDF."""
    with open(Path(__file__).parent / "sample_pdf" / "sample_pdf.txt", "r") as f:
        return f.read().strip()


@pytest.fixture
def sample_pptx():
    """Get the sample PowerPoint file."""
    return Path(__file__).parent / "sample_pptx" / "sample.pptx"


@pytest.fixture
def sample_pptx_text():
    """Get the expected text from the sample PowerPoint."""
    with open(Path(__file__).parent / "sample_pptx" / "sample_pptx.txt", "r") as f:
        return f.read().strip()


@pytest.fixture
def learning_resources_dir(temp_dir):
    """Create a Learning Resources directory with sample files."""
    resources_dir = temp_dir / "Learning Resources"
    resources_dir.mkdir()
    return resources_dir


def test_extract_text_from_pdf(sample_pdf, sample_pdf_text):
    """Test PDF text extraction."""
    text = extract_text_from_pdf(sample_pdf)
    # Check if the extracted text contains key phrases from the sample
    assert any(phrase in text for phrase in sample_pdf_text.split("\n"))


def test_extract_text_from_pdf_invalid_file():
    """Test PDF text extraction with invalid file."""
    text = extract_text_from_pdf(Path("nonexistent.pdf"))
    assert text == ""


def test_extract_text_from_pptx(sample_pptx, sample_pptx_text):
    """Test PowerPoint text extraction."""
    text = extract_text_from_pptx(sample_pptx)
    # Check if the extracted text contains key phrases from the sample
    assert any(phrase in text for phrase in sample_pptx_text.split("\n"))


def test_extract_text_from_pptx_invalid_file():
    """Test PowerPoint text extraction with invalid file."""
    text = extract_text_from_pptx(Path("nonexistent.pptx"))
    assert text == ""


def test_combine_inputs_to_text(
    learning_resources_dir, sample_pdf, sample_pptx, sample_pdf_text, sample_pptx_text
):
    """Test combining inputs from different file types."""
    # Copy sample files to resources directory
    shutil.copy(sample_pdf, learning_resources_dir)
    shutil.copy(sample_pptx, learning_resources_dir)

    text = combine_inputs_to_text(learning_resources_dir)

    # Check if the combined text contains content from both files
    assert any(phrase in text for phrase in sample_pdf_text.split("\n"))
    assert any(phrase in text for phrase in sample_pptx_text.split("\n"))


def test_create_obsidian_notes(temp_dir):
    """Test note creation."""
    topics = """
(Test1.md)
---
# Test 1
Content 1
---
(Test2.md)
---
# Test 2
Content 2
---
"""
    create_obsidian_notes(topics, temp_dir)

    # Check if files were created
    assert (temp_dir / "Test1.md").exists()
    assert (temp_dir / "Test2.md").exists()

    # Check file contents
    with open(temp_dir / "Test1.md", "r") as f:
        assert "# Test 1" in f.read()
    with open(temp_dir / "Test2.md", "r") as f:
        assert "# Test 2" in f.read()


def test_create_obsidian_notes_interactive(temp_dir, monkeypatch):
    """Test interactive note creation."""
    topics = """
(Test1.md)
---
# Test 1
Content 1
---
"""
    # Mock user input
    monkeypatch.setattr("builtins.input", lambda _: "y")

    create_obsidian_notes(topics, temp_dir, interactive=True)
    assert (temp_dir / "Test1.md").exists()


def test_setup_logging(temp_dir):
    """Test logging setup."""
    log_file = temp_dir / "test.log"
    setup_logging(verbosity=1, log_file=str(log_file))

    # Test different log levels
    logging.info("Test info")
    logging.error("Test error")
    logging.debug("Test debug")

    # Check log file contents
    with open(log_file, "r") as f:
        log_content = f.read()
        assert "Test info" in log_content
        assert "Test error" in log_content
        assert "Test debug" not in log_content  # Debug not shown in v1


@patch("gen_notes.check_dependencies")
@patch("gen_notes.combine_inputs_to_text")
@patch("gen_notes.create_obsidian_notes")
def test_main(mock_create, mock_combine, mock_check, temp_dir):
    """Test main function execution."""
    # Mock the necessary functions
    mock_combine.return_value = "Test content"

    # Create test directory structure
    resources_dir = temp_dir / "Learning Resources"
    resources_dir.mkdir()

    # Run main with test arguments
    with patch("sys.argv", ["gen_notes.py", str(temp_dir), "-v1"]):
        main()

    # Verify function calls
    mock_check.assert_called_once()
    mock_combine.assert_called_once()
    mock_create.assert_called_once()


def test_main_invalid_directory():
    """Test main function with invalid directory."""
    with patch("sys.argv", ["gen_notes.py", "/nonexistent/directory"]):
        with pytest.raises(SystemExit):
            main()
