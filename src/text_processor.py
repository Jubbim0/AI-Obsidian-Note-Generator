"""Text processor class for handling text processing operations."""

from .logging_helper import LoggingHelper
from .openai_helper import OpenAIHelper


class TextProcessor:
    """Handles text processing operations."""

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
        self.logger = LoggingHelper(verbosity)
        self.openai = OpenAIHelper()
