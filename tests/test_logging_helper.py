"""Tests for the logging helper module."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from tqdm import tqdm
from src.logging_helper import (
    setup_colored_logging,
    create_progress_bar,
    update_progress_bar,
    close_progress_bar,
    Colors,
    ColoredFormatter,
)


def test_setup_colored_logging():
    """Test that colored logging is set up correctly."""
    # Test silent mode
    setup_colored_logging(0)
    assert logging.root.level == logging.ERROR

    # Test normal mode
    setup_colored_logging(1)
    assert logging.root.level == logging.INFO

    # Test verbose mode
    setup_colored_logging(2)
    assert logging.root.level == logging.DEBUG

    # Test with log file
    with patch("logging.FileHandler") as mock_file_handler:
        setup_colored_logging(1, "test.log")
        mock_file_handler.assert_called_once_with("test.log")


def test_create_progress_bar():
    """Test progress bar creation with different verbosity levels."""
    # Test with verbosity 1 (should create progress bar)
    pbar = create_progress_bar(100, "Test", 1)
    assert isinstance(pbar, tqdm)
    assert pbar.total == 100
    assert pbar.desc == "Test"
    pbar.close()

    # Test with verbosity 0 (should return None)
    pbar = create_progress_bar(100, "Test", 0)
    assert pbar is None


def test_update_progress_bar():
    """Test progress bar updates."""
    # Test with valid progress bar
    pbar = create_progress_bar(100, "Test", 1)
    with patch.object(pbar, "update") as mock_update:
        update_progress_bar(pbar, 50, 100)
        mock_update.assert_called_once()
    pbar.close()

    # Test with None progress bar
    update_progress_bar(None, 50, 100)  # Should not raise any errors


def test_close_progress_bar():
    """Test progress bar closing."""
    # Test with valid progress bar
    pbar = create_progress_bar(100, "Test", 1)
    with patch.object(pbar, "close") as mock_close:
        close_progress_bar(pbar)
        mock_close.assert_called_once()

    # Test with None progress bar
    close_progress_bar(None)  # Should not raise any errors


def test_colored_formatter():
    """Test that the colored formatter adds colors correctly."""
    formatter = ColoredFormatter("%(message)s")

    # Create a mock record for each log level
    for level, color in ColoredFormatter.COLORS.items():
        record = logging.LogRecord(
            name="test",
            level=getattr(logging, level),
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert color in formatted
        assert Colors.ENDC in formatted


def test_progress_bar_integration():
    """Test that progress bars work correctly in an integration scenario."""
    # Create a progress bar
    pbar = create_progress_bar(100, "Integration Test", 1)

    # Simulate progress updates
    for i in range(0, 101, 20):
        update_progress_bar(pbar, i, 100)

    # Close the progress bar
    close_progress_bar(pbar)

    # Verify the progress bar was updated correctly
    assert pbar.n == 100  # Should be at 100%
