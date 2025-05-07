"""Logging helper module for AI Note Workflow."""

import logging
from typing import Optional
from tqdm import tqdm
import sys


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""

    COLORS = {
        "DEBUG": Colors.CYAN,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "CRITICAL": Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}{Colors.ENDC}"
        return super().format(record)


def setup_colored_logging(verbosity: int, log_file: Optional[str] = None):
    """Set up logging with colors and optional file output."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set root logger level based on verbosity
    if verbosity == 0:
        logging.root.setLevel(logging.ERROR)
    elif verbosity == 1:
        logging.root.setLevel(logging.INFO)
    elif verbosity == 2:
        logging.root.setLevel(logging.DEBUG)

    # Create formatters
    colored_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set up console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.root.level)
    console_handler.setFormatter(colored_formatter)
    logging.root.addHandler(console_handler)

    # Set up file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.root.level)
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


def create_progress_bar(total: int, desc: str, verbosity: int = 1) -> Optional[tqdm]:
    """Create a progress bar if verbosity level is appropriate."""
    if verbosity >= 1:
        return tqdm(
            total=total,
            desc=desc,
            unit="%",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    return None


def update_progress_bar(pbar: Optional[tqdm], current: int, total: int):
    """Update progress bar if it exists."""
    if pbar is not None:
        progress = min(100, int((current / total) * 100))
        pbar.update(progress - pbar.n)


def close_progress_bar(pbar: Optional[tqdm]):
    """Close progress bar if it exists."""
    if pbar is not None:
        pbar.update(100 - pbar.n)
        pbar.close()


class LoggingHelper:
    """Handles all logging operations with consistent formatting."""

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity

    def log_section_header(self, title: str, width: int = 80) -> None:
        """Log a section header with visual separators."""
        separator = "=" * width
        logging.info(f"\n{separator}")
        logging.info(title)
        logging.info(f"{separator}\n")

    def log_subsection(self, title: str, width: int = 40) -> None:
        """Log a subsection header with visual separators."""
        logging.info(f"\n{title}:")
        logging.info("-" * width)

    def log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, cost: float, model: str = None
    ) -> None:
        """Log token usage and cost information."""
        self.log_subsection("Token Usage")
        if model:
            logging.info(f"Model: {model}")
        logging.info(f"Prompt tokens: {prompt_tokens}")
        logging.info(f"Completion tokens: {completion_tokens}")
        logging.info(f"Total cost: ${cost:.4f}\n")

    def log_processing_summary(
        self, duplicates_found: int, original_count: int, unique_count: int
    ) -> None:
        """Log a summary of the processing results."""
        self.log_subsection("Processing Summary")
        logging.info(f"Duplicates merged: {duplicates_found}")
        logging.info(f"Original topics: {original_count}")
        logging.info(f"Unique topics: {unique_count}\n")

        if self.verbosity >= 2:
            logging.debug("\nFinal Topic Structure:")
            logging.debug("-" * 40)

    def log_duplicate_found(
        self, title: str, original_length: int, new_length: int, combined_length: int
    ) -> None:
        """Log information about a found duplicate."""
        self.log_subsection("Duplicate Found")
        logging.info(f"Topic: {title}")
        logging.info(f"Original content length: {original_length} chars")
        logging.info(f"New content length: {new_length} chars")
        logging.info(f"Combined content length: {combined_length} chars\n")
