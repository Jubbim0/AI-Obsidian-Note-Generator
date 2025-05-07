"""OpenAI helper for AI Note Workflow."""

import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from .logging_helper import create_progress_bar, update_progress_bar, close_progress_bar


class OpenAIHelper:
    """Helper class for OpenAI API interactions."""

    def __init__(
        self, client: Optional[OpenAI] = None, system_prompt: Optional[str] = None
    ):
        """Initialize the OpenAI helper."""
        self.client = client or OpenAI()
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.documents: List[str] = []
        self.verbosity = 1

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for topic extraction."""
        return """You are an expert at analyzing educational content and extracting key topics and their relationships.
        For the given content, identify main topics and their subtopics.
        Return the response in the following JSON format:
        {
            "topics": {
                "Main Topic 1": {
                    "Subtopic 1": "Description",
                    "Subtopic 2": "Description"
                },
                "Main Topic 2": {
                    "Subtopic 1": "Description",
                    "Subtopic 2": "Description"
                }
            },
            "associated_topics": {
                "Main Topic 1": ["Related Topic 1", "Related Topic 2"],
                "Main Topic 2": ["Related Topic 3", "Related Topic 4"]
            }
        }"""

    def add_document(self, text: str):
        """Add a document to be processed."""
        self.documents.append(text)

    def generate_topics(self, verbosity: int = 1) -> Dict[str, str]:
        """Generate topics from all added documents."""
        if not self.documents:
            logging.warning("No documents to process")
            return {}

        self.verbosity = verbosity
        logging.info("Generating topics from documents...")

        # Create progress bar for topic generation
        pbar = create_progress_bar(len(self.documents), "Generating topics", verbosity)

        all_topics = {"topics": {}, "associated_topics": {}}
        total_cost = 0.0

        for i, doc in enumerate(self.documents):
            try:
                # Process content with GPT-4
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": doc},
                    ],
                    temperature=0.4,
                )

                # Calculate cost
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                cost = self.calculate_token_cost(
                    "gpt-4", prompt_tokens, completion_tokens
                )
                total_cost += cost

                if verbosity >= 1:
                    logging.info(
                        "Document {}/{} processed".format(i + 1, len(self.documents))
                    )
                    logging.info("Cost: ${:.4f}".format(cost))

                # Parse the JSON response
                try:
                    content = response.choices[0].message.content
                    if isinstance(content, str):
                        topics = json.loads(content)
                        self._merge_topics(all_topics, topics)
                    else:
                        logging.error("Invalid response format")
                except json.JSONDecodeError as e:
                    logging.error("Error parsing topics JSON: {}".format(e))

            except Exception as e:
                logging.error("Error processing document: {}".format(e))

            update_progress_bar(pbar, i + 1, len(self.documents))

        close_progress_bar(pbar)

        if verbosity >= 1:
            logging.info("Total cost for all documents: ${:.4f}".format(total_cost))

        return all_topics

    def calculate_token_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate the cost of token usage.

        Args:
            model (str): The model used
            prompt_tokens (int): Number of prompt tokens
            completion_tokens (int): Number of completion tokens

        Returns:
            float: The cost in USD
        """
        # GPT-4 pricing (as of 2024)
        if model == "gpt-4":
            return (prompt_tokens * 0.03 / 1000) + (completion_tokens * 0.06 / 1000)
        # Default to 0 for unknown models
        return 0.0

    def _merge_topics(self, all_topics: Dict, new_topics: Dict) -> None:
        """Merge topics from a document into the main topics dictionary."""
        if not new_topics.get("topics"):
            return

        # Merge topics and subtopics
        for topic, subtopics in new_topics["topics"].items():
            if topic not in all_topics["topics"]:
                all_topics["topics"][topic] = subtopics
            else:
                # Merge subtopics
                self._merge_subtopics(all_topics["topics"][topic], subtopics)

        # Merge associated topics
        if "associated_topics" in new_topics:
            for topic, associated in new_topics["associated_topics"].items():
                if topic not in all_topics["associated_topics"]:
                    all_topics["associated_topics"][topic] = []
                all_topics["associated_topics"][topic].extend(associated)
                # Remove duplicates while preserving order
                all_topics["associated_topics"][topic] = list(
                    dict.fromkeys(all_topics["associated_topics"][topic])
                )

    def _merge_subtopics(self, existing: Dict, new: Dict) -> None:
        """Recursively merge subtopics."""
        for subtopic, content in new.items():
            if subtopic not in existing:
                existing[subtopic] = content
            elif isinstance(content, dict) and isinstance(existing[subtopic], dict):
                self._merge_subtopics(existing[subtopic], content)
