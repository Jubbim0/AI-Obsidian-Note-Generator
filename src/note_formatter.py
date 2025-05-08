"""Note formatter class for handling note formatting and Wikipedia definitions."""

import logging
import urllib.parse
import requests
from typing import Dict, List, Set
from pathlib import Path

from .logging_helper import LoggingHelper


class NoteFormatter:
    """Handles formatting of notes based on topic structure."""

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
        self.logger = LoggingHelper(verbosity)
        self._existing_topics: Set[str] = set()

    def set_existing_topics(self, topics: Dict[str, Dict]) -> None:
        """Set the list of existing topics that will be created.

        Args:
            topics (Dict[str, Dict]): Dictionary of topics and their subtopics
        """
        self._existing_topics = set(topics.keys())

    def get_wikipedia_definition(self, topic: str) -> str:
        """Get the definition of a topic from Wikipedia."""
        try:
            # First search for the topic to find the closest match
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(topic)}&format=json"
            search_response = requests.get(search_url)
            search_response.raise_for_status()
            search_data = search_response.json()

            # If no search results found, return blank definition
            if not search_data.get("query", {}).get("search"):
                return ""

            # Get the title of the closest match
            closest_match = search_data["query"]["search"][0]["title"]

            # Now get the summary for the closest match
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(closest_match)}"
            summary_response = requests.get(summary_url)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            return summary_data.get("extract", "")

        except Exception as e:
            self.logger.log_subsection("Wikipedia API Error")
            logging.error(f"Error fetching definition for {topic}: {e}")
            return ""

    def format_topic_content(
        self, topic: str, subtopics: Dict, associated_topics: List[str]
    ) -> str:
        """Format the content for a topic file."""
        # Get Wikipedia definition
        definition = self.get_wikipedia_definition(topic)

        # Start building the content
        content = [f"# {topic}", "", "## Definition", f"*{definition}*", ""]

        # Add subtopics with proper heading levels
        self._add_subtopics(content, subtopics, 2)

        # Filter associated topics to only include those that exist or will be created
        valid_associated_topics = [
            topic for topic in associated_topics if topic in self._existing_topics
        ]

        # Add Related Topics section only if there are valid associated topics
        if valid_associated_topics:
            content.extend(
                [
                    "",
                    "## Related Topics",
                    *[
                        f"- [{topic}]({urllib.parse.quote(topic)}.md)"
                        for topic in valid_associated_topics
                    ],
                ]
            )

        return "\n".join(content)

    def _add_subtopics(self, content: List[str], subtopics: Dict, level: int) -> None:
        """Recursively add subtopics with proper heading levels."""
        for subtopic, nested in subtopics.items():
            # Add the current subtopic
            content.append(f"{'#' * level} {subtopic}")
            content.append("")

            # If there are nested subtopics, add them
            if isinstance(nested, dict):
                self._add_subtopics(content, nested, level + 1)
            elif isinstance(nested, str):
                # If it's a string, it's the content
                content.append(nested)
                content.append("")
