"""Document processor for AI Note Workflow."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.openai_helper import OpenAIHelper
from src.utils import (
    extract_text_from_pdf,
    extract_text_from_pptx,
    transcribe_audio_with_whisper,
    extract_text_from_txt,
)
from src.logging_helper import LoggingHelper
from .logging_helper import create_progress_bar, update_progress_bar, close_progress_bar


class DocumentProcessor:
    """Process documents and extract topics using OpenAI."""

    def __init__(self, verbosity=1, openai_helper=None, logger=None):
        """Initialize the document processor.

        Args:
            verbosity (int): Logging verbosity level (0-2)
            openai_helper (OpenAIHelper, optional): OpenAI helper instance
            logger (LoggingHelper, optional): Logging helper instance
        """
        self.verbosity = verbosity
        self.openai_helper = openai_helper or OpenAIHelper()
        self.logger = logger or LoggingHelper(verbosity=verbosity)
        self.existing_topics = set()

    def process_directory(
        self, directory: Path
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Process all documents in a directory and extract topics.

        Args:
            directory (Path): Path to the directory containing documents

        Returns:
            Dict[str, str]: Extracted topics
        """
        logging.info(f"Processing directory: {directory}")

        # Get all files in directory by type
        pdf_files = list(directory.glob("*.pdf"))
        pptx_files = list(directory.glob("*.pptx")) + list(directory.glob("*.ppt"))
        txt_files = list(directory.glob("*.txt"))
        audio_files = (
            list(directory.glob("*.mp4"))
            + list(directory.glob("*.wav"))
            + list(directory.glob("*.mp3"))
        )

        total_files = (
            len(pdf_files) + len(pptx_files) + len(txt_files) + len(audio_files)
        )
        if total_files == 0:
            logging.warning("No supported files found in directory")
            return {}

        # Create progress bar for document processing
        pbar = create_progress_bar(total_files, "Processing documents", self.verbosity)
        processed = 0

        # Process PDFs
        for file_path in pdf_files:
            try:
                logging.info(f"Processing PDF: {file_path.name}")
                text = extract_text_from_pdf(file_path, self.verbosity)
                if text:
                    self.openai_helper.add_document(text, file_path.stem)
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
            processed += 1
            update_progress_bar(pbar, processed, total_files)

        # Process PowerPoint files
        for file_path in pptx_files:
            try:
                logging.info(f"Processing PowerPoint: {file_path.name}")
                text = extract_text_from_pptx(file_path, self.verbosity)
                if text:
                    self.openai_helper.add_document(text, file_path.stem)
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
            processed += 1
            update_progress_bar(pbar, processed, total_files)

        # Process text files
        for file_path in txt_files:
            try:
                logging.info(f"Processing text file: {file_path.name}")
                text = extract_text_from_txt(file_path, self.verbosity)
                if text:
                    self.openai_helper.add_document(text, file_path.stem)
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
            processed += 1
            update_progress_bar(pbar, processed, total_files)

        # Process audio files
        for file_path in audio_files:
            try:
                logging.info(f"Transcribing {file_path.name} using Whisper...")
                text = transcribe_audio_with_whisper(file_path, self.verbosity)
                if text:
                    self.openai_helper.add_document(text, file_path.stem)
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
            processed += 1
            update_progress_bar(pbar, processed, total_files)

        close_progress_bar(pbar)

        # Generate topics from processed documents
        logging.info("Generating topics from processed documents...")
        topics = self.openai_helper.generate_topics()

        return topics, self.openai_helper.index_topics

    def process_single_document(self, file_path: Path) -> Tuple[Dict[str, str], float]:
        """Process a single document and extract topics.

        Args:
            file_path (Path): Path to the document file

        Returns:
            Tuple[Dict[str, str], float]: Topics and cost
        """
        # Extract text based on file type
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path, self.verbosity)
        elif file_path.suffix.lower() in [".pptx", ".ppt"]:
            text = extract_text_from_pptx(file_path, self.verbosity)
        elif file_path.suffix.lower() in [".mp4", ".wav", ".mp3"]:
            text = transcribe_audio_with_whisper(file_path, self.verbosity)
        else:
            self.logger.log_warning(f"Unsupported file type: {file_path.suffix}")
            return {}, 0.0

        # Add document to OpenAI helper
        self.openai_helper.add_document(text, file_path.stem)

        # Generate topics
        topics = self.openai_helper.generate_topics(self.verbosity)

        return topics, 0.0  # Cost is tracked internally by OpenAIHelper

    def _merge_topics(self, all_topics: Dict, new_topics: Dict) -> None:
        """Merge new topics into existing topics.

        Args:
            all_topics (Dict): Existing topics to merge into
            new_topics (Dict): New topics to merge
        """
        # Merge main topics
        for topic, subtopics in new_topics.get("topics", {}).items():
            if topic not in all_topics["topics"]:
                all_topics["topics"][topic] = {}
            all_topics["topics"][topic].update(subtopics)

        # Merge associated topics
        for topic, associated in new_topics.get("associated_topics", {}).items():
            if topic not in all_topics["associated_topics"]:
                all_topics["associated_topics"][topic] = []
            all_topics["associated_topics"][topic].extend(associated)
            # Remove duplicates
            all_topics["associated_topics"][topic] = list(
                set(all_topics["associated_topics"][topic])
            )
