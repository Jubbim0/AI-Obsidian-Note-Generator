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
from .logging_helper import LoggingHelper, setup_colored_logging
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
    print(f"Verbosity: {args.verbosity}")
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
    topics, index_topics = doc_processor.process_directory(resource_path)
    if not topics:
        logging.error("No topics were generated.")
        sys.exit(1)

    # Set existing topics in the note formatter
    note_formatter.set_existing_topics(topics["topics"])

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

            if args.interactive:
                print(f"\nFile to be created: {note_file.name}")
                while True:
                    response = input(
                        "Create this file? (y)es/(n)o/(e)dit/(a)bort: "
                    ).lower()
                    if response == "y":
                        break
                    elif response == "n":
                        logging.info(f"Skipping file: {note_file.name}")
                        continue
                    elif response == "e":
                        # Create a temporary file with the content
                        import tempfile
                        import subprocess
                        import os

                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".md", delete=False
                        ) as temp_file:
                            temp_file.write(content)
                            temp_path = temp_file.name

                        try:
                            # Open the file in nano
                            subprocess.run(["nano", temp_path], check=True)

                            # Read the edited content
                            with open(temp_path, "r") as temp_file:
                                content = temp_file.read()

                            # Clean up the temporary file
                            os.unlink(temp_path)
                            break
                        except Exception as e:
                            logging.error(f"Error editing file: {e}")
                            os.unlink(temp_path)
                            continue
                    elif response == "a":
                        logging.info("Aborting all file creation")
                        return
                    else:
                        print("Invalid input. Please enter y, n, e, or a.")

            # Write the content to the file
            with open(note_file, "w", encoding="utf-8") as f:
                f.write(content)

            if args.verbosity >= 1:
                logging.info(f"Created note: {note_file.name}")

        except Exception as e:
            logging.error(f"Error creating note for {topic}: {e}")
            continue

    for file_name, subtopics in index_topics.items():
        if not subtopics:
            continue
        try:
            content = note_formatter.format_index_page(subtopics)
            note_file = root_path / f"{file_name}-Index.md"
            with open(note_file, "w", encoding="utf-8") as f:
                f.write(content)

            if args.verbosity >= 1:
                logging.info(f"Created note: {note_file.name}")

        except Exception as e:
            logging.error(f"Error creating note for {file_name}: {e}")
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
