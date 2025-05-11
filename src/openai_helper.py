"""OpenAI helper for AI Note Workflow."""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from .logging_helper import create_progress_bar, update_progress_bar, close_progress_bar
import openai

# Constants for rate limiting
MAX_TOKENS_PER_MINUTE = 30000  # GPT-4 Turbo's limit
TOKEN_WINDOW = 60  # 1 minute window
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # Initial backoff in seconds


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
        self.token_usage_history: List[Tuple[float, int]] = []  # (timestamp, tokens)
        self.chunk_size = 100000  # Approximate tokens per chunk

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

    def _clean_token_history(self):
        """Remove token usage records older than the window."""
        current_time = time.time()
        self.token_usage_history = [
            (ts, tokens)
            for ts, tokens in self.token_usage_history
            if current_time - ts < TOKEN_WINDOW
        ]

    def _get_current_token_usage(self) -> int:
        """Get the total tokens used in the current window."""
        self._clean_token_history()
        return sum(tokens for _, tokens in self.token_usage_history)

    def _wait_for_token_limit(self, required_tokens: int) -> None:
        """Wait if necessary to stay within token rate limits."""
        while True:
            current_usage = self._get_current_token_usage()
            if current_usage + required_tokens <= MAX_TOKENS_PER_MINUTE:
                break
            time.sleep(1)  # Wait a second and check again

    def _chunk_document(self, text: str) -> List[str]:
        """Split a document into smaller chunks based on approximate token count."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        chunks = []
        current_chunk = ""
        current_size = 0

        # Split by paragraphs to maintain context
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            # Rough estimate of tokens in this paragraph
            paragraph_tokens = len(paragraph) // 4

            if current_size + paragraph_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
                current_size = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                current_size += paragraph_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _make_api_call_with_retry(
        self, messages: List[Dict], temperature: float = 0.4
    ) -> Dict:
        """Make an API call with exponential backoff retry logic and granular error handling."""
        backoff = INITIAL_BACKOFF
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # Estimate tokens in the request
                estimated_tokens = sum(len(msg["content"]) // 4 for msg in messages)
                self._wait_for_token_limit(estimated_tokens)

                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=temperature,
                )

                # Record token usage
                total_tokens = (
                    response.usage.prompt_tokens + response.usage.completion_tokens
                )
                self.token_usage_history.append((time.time(), total_tokens))

                return response

            except openai.error.RateLimitError as e:
                logging.warning(f"OpenAI rate limit exceeded: {e}")
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logging.error("Max retries reached for rate limit error.")
            except openai.error.AuthenticationError as e:
                logging.error(f"OpenAI authentication error: {e}. Check your API key.")
                raise
            except openai.error.APIConnectionError as e:
                logging.error(
                    f"OpenAI API connection error: {e}. Check your network connection."
                )
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logging.error("Max retries reached for API connection error.")
            except openai.error.Timeout as e:
                logging.error(f"OpenAI request timed out: {e}.")
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logging.error("Max retries reached for timeout error.")
            except openai.error.InvalidRequestError as e:
                logging.error(
                    f"OpenAI invalid request: {e}. Check your prompt and parameters."
                )
                raise
            except openai.error.APIError as e:
                logging.error(
                    f"OpenAI server error: {e}. This may be a temporary issue."
                )
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logging.error("Max retries reached for server error.")
            except Exception as e:
                logging.error(f"Unexpected error from OpenAI: {e}")
                last_error = e
                raise

        raise last_error

    def add_document(self, text: str):
        """Add a document to be processed."""
        self.documents.append(text)

    def _clean_json_response(self, content: str) -> str:
        """Clean a JSON response by removing markdown formatting and extra characters.

        Args:
            content (str): The raw response content

        Returns:
            str: Cleaned JSON string
        """
        # Find the first '{' and last '}'
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No valid JSON object found in response")

        # Extract just the JSON object
        return content[start : end + 1]

    def generate_topics(self, verbosity: int = 1) -> Dict[str, str]:
        """Generate topics from all added documents."""
        if not self.documents:
            logging.warning("No documents to process")
            return {}

        self.verbosity = verbosity
        logging.info("Generating topics from documents...")

        all_topics = {"topics": {}, "associated_topics": {}}
        total_cost = 0.0

        for i, doc in enumerate(self.documents):
            try:
                # Split document into chunks
                chunks = self._chunk_document(doc)
                chunk_topics = []

                for chunk in chunks:
                    # Process each chunk
                    response = self._make_api_call_with_retry(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": chunk},
                        ]
                    )

                    # Calculate cost
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(response.usage, "completion_tokens", 0)
                    cost = self.calculate_token_cost(
                        "gpt-4-turbo-preview", prompt_tokens, completion_tokens
                    )
                    total_cost += cost

                    if verbosity >= 1:
                        logging.info(
                            "Chunk processed for document {}/{}".format(
                                i + 1, len(self.documents)
                            )
                        )
                        logging.info("Cost: ${:.4f}".format(cost))

                    # Parse the JSON response
                    try:
                        content = response.choices[0].message.content
                        if isinstance(content, str):
                            cleaned_content = self._clean_json_response(content)
                            topics = json.loads(cleaned_content)
                            chunk_topics.append(topics)
                        else:
                            logging.error("Invalid response format")
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error("Error parsing topics JSON: {}".format(e))
                        if verbosity >= 2:
                            logging.debug("Raw response content: {}".format(content))

                # Merge topics from all chunks
                for topics in chunk_topics:
                    self._merge_topics(all_topics, topics)

            except Exception as e:
                logging.error("Error processing document: {}".format(e))

        if verbosity >= 1:
            logging.info("Total cost for all documents: ${:.4f}".format(total_cost))

        # Print summary at the end
        self.print_summary()

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
        # GPT-4 Turbo pricing (as of 2024)
        if model == "gpt-4-turbo-preview":
            return (prompt_tokens * 0.01 / 1000) + (completion_tokens * 0.03 / 1000)
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

    def print_summary(self):
        """Print a summary of statistics at the end of the program."""
        total_cost = (
            sum(self.token_usage_history, key=lambda x: x[1])[1] * 0.01 / 1000
        )  # Assuming 0.01 per 1K tokens
        num_docs_processed = len(self.documents)
        num_docs_made = len(
            self.documents
        )  # Assuming each document is processed into a note
        total_time = (
            sum(self.token_usage_history, key=lambda x: x[0])[0]
            - self.token_usage_history[0][0]
            if self.token_usage_history
            else 0
        )

        print("\n\n\n=== Summary ===")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Number of documents processed: {num_docs_processed}")
        print(f"Number of documents made: {num_docs_made}")
        print(f"Time taken for API calls: {total_time:.2f} seconds")
        print("==============\n")
