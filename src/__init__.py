"""
AI Note Workflow package for generating structured notes from various learning resources.
"""

import sys
import argparse
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .document_processor import DocumentProcessor
from .note_formatter import NoteFormatter
from .logging_helper import LoggingHelper
from .openai_helper import OpenAIHelper
from .utils import setup_logging, check_dependencies


def create_processors(verbosity: int = 1):
    """Create processor instances with dependencies."""
    logger = LoggingHelper(verbosity)
    openai_helper = OpenAIHelper()
    doc_processor = DocumentProcessor(
        verbosity, openai_helper=openai_helper, logger=logger
    )
    note_formatter = NoteFormatter(verbosity)
    return doc_processor, note_formatter, logger


def main():
    """Main function to process documents and create notes."""
    parser = argparse.ArgumentParser(
        description="Generate structured notes from various learning resources."
    )
    parser.add_argument("resource_path", help="Path to the resource folder")
    parser.add_argument(
        "-v0", action="store_const", const=0, dest="verbosity", help="Silent mode"
    )
    parser.add_argument(
        "-v1",
        action="store_const",
        const=1,
        dest="verbosity",
        help="Basic progress output",
    )
    parser.add_argument(
        "-v2", action="store_const", const=2, dest="verbosity", help="Detailed output"
    )
    parser.add_argument("-l", "--log-file", help="Log to file")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Interactive mode"
    )
    parser.set_defaults(verbosity=1)

    args = parser.parse_args()
    setup_logging(args.verbosity, args.log_file)

    # Check dependencies first
    check_dependencies()

    # Convert to Path object
    resource_path = Path(args.resource_path)
    # Append "Learning Resources" to the path
    resource_path = resource_path / "Learning Resources"
    root_path = resource_path.parent

    if not resource_path.exists():
        logging.error(f"{resource_path} does not exist.")
        sys.exit(1)

    # Initialize processors with dependencies
    doc_processor, note_formatter, logger = create_processors(args.verbosity)

    logger.log_section_header("Processing Documents")

    # Process all documents and get combined topics
    topics = doc_processor.process_directory(resource_path)
    if not topics:
        logging.error("No topics were generated.")
        sys.exit(1)

    # Create notes directory
    notes_dir = root_path / "notes"
    notes_dir.mkdir(exist_ok=True)

    # Create notes for each topic
    logger.log_section_header("Creating Notes")
    for topic, subtopics in topics["topics"].items():
        try:
            # Format the note content
            content = note_formatter.format_topic_content(
                topic, subtopics, topics["associated_topics"].get(topic, [])
            )

            # Create the note file
            note_file = notes_dir / f"{topic}.md"
            with open(note_file, "w", encoding="utf-8") as f:
                f.write(content)

            if args.verbosity >= 1:
                logging.info(f"Created note: {note_file.name}")

        except Exception as e:
            logging.error(f"Error creating note for {topic}: {e}")
            continue

    logging.info("Done!")


from .openai_helper import OpenAIHelper
from .text_processor import TextProcessor
from .document_processor import DocumentProcessor
from .topic_processor import TopicProcessor
from .logging_helper import LoggingHelper
from .note_formatter import NoteFormatter
from .utils import (
    setup_logging,
    check_dependencies,
    extract_text_from_pdf,
    extract_text_from_pptx,
    transcribe_audio_with_whisper,
)

__version__ = "1.0.0"
