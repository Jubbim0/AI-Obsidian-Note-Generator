"""Topic processor class for handling topic processing and structure generation."""

import json
import logging
from typing import Dict, List

from .logging_helper import LoggingHelper
from .openai_helper import OpenAIHelper


class TopicProcessor:
    """Handles topic processing and structure generation."""

    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
        self.logger = LoggingHelper(verbosity)
        self.openai = OpenAIHelper()

    def process_topics(self, topics: str) -> Dict:
        """Process topics into a structured JSON format."""
        self.logger.log_section_header("Processing Topics")

        try:
            # Parse the topics string into a structured format
            topic_structure = {}
            associated_topics = {}
            current_topic = None
            current_subtopics = {}
            current_level = 0
            current_path = []

            lines = topics.split("\n")
            for line in lines:
                line = line.strip()
                if not line or line == "---":
                    continue

                # Count leading # to determine level
                level = 0
                while line.startswith("#"):
                    level += 1
                    line = line[1:].strip()

                if level == 0:
                    # This is a main topic
                    if current_topic:
                        topic_structure[current_topic] = current_subtopics
                    current_topic = line
                    current_subtopics = {}
                    current_path = [current_topic]
                    associated_topics[current_topic] = []
                else:
                    # This is a subtopic
                    if level <= current_level:
                        # Go back up the tree
                        current_path = current_path[: level - 1]

                    current_path.append(line)
                    current_level = level

                    # Add to the structure
                    current = topic_structure
                    for i, path in enumerate(current_path[:-1]):
                        if path not in current:
                            current[path] = {}
                        current = current[path]
                    current[current_path[-1]] = {}

            # Add the last topic
            if current_topic:
                topic_structure[current_topic] = current_subtopics

            # Get associated topics from the LLM
            for topic in topic_structure.keys():
                associated = self._get_associated_topics(
                    topic, list(topic_structure.keys())
                )
                associated_topics[topic] = associated

            return {"topics": topic_structure, "associated_topics": associated_topics}

        except Exception as e:
            self.logger.log_subsection("Error Processing Topics")
            logging.error(f"Error processing topics: {e}")
            return {"topics": {}, "associated_topics": {}}

    def _get_associated_topics(self, topic: str, all_topics: List[str]) -> List[str]:
        """Get associated topics for a given topic using GPT-3.5-turbo."""
        try:
            prompt = f"""Given the topic "{topic}" and the following list of topics:
{", ".join(all_topics)}

Return a JSON array of the 3 most closely related topics from the list. Only include topics that are actually in the list."""

            response = self.openai.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that identifies related topics.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            # Parse the response and ensure it's a valid list of topics
            try:
                associated = json.loads(response.choices[0].message.content)
                # Filter to ensure only valid topics are included
                return [t for t in associated if t in all_topics][:3]
            except:
                return []

        except Exception as e:
            logging.error(f"Error getting associated topics for {topic}: {e}")
            return []
