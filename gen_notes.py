import sys
import subprocess
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
import logging

# ============ CONFIGURATION ============
# Load environment variables first
load_dotenv()

# Get API key and validate
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please create a .env file with your API key."
    )

# Initialize OpenAI client after validating API key
client = OpenAI(api_key=api_key)


# ============ LOGGING SETUP ============
def setup_logging(verbosity: int, log_file: str = None):
    """Set up logging based on verbosity level and optional log file."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Set root logger level based on verbosity
    if verbosity == 0:
        logging.root.setLevel(logging.ERROR)
    elif verbosity == 1:
        logging.root.setLevel(logging.INFO)
    elif verbosity == 2:
        logging.root.setLevel(logging.DEBUG)

    # Set up console handler with the same level as root logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.root.level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)

    # Set up file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.root.level)  # Use same level as root logger
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


# ============ FILE HANDLERS ============


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pypdf
        import pptx
    except ImportError as e:
        logging.error(f"Required dependency not installed: {e}")
        logging.info("Please install required packages: pip install pypdf python-pptx")
        sys.exit(1)


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path.name}: {e}")
        return ""


def extract_text_from_pptx(pptx_path: Path) -> str:
    try:
        from pptx import Presentation

        prs = Presentation(str(pptx_path))
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs)
    except Exception as e:
        logging.error(f"Error processing PPTX {pptx_path.name}: {e}")
        return ""


def transcribe_audio_with_whisper(mp4_path: Path, verbosity: int = 1) -> str:
    try:
        logging.info(f"Transcribing {mp4_path.name} using Whisper...")

        # Capture output if verbosity is not 2
        capture_output = verbosity < 2

        result = subprocess.run(
            [
                "whisper",
                str(mp4_path),
                "--model",
                "base",
                "--language",
                "en",
                "--output_format",
                "txt",
            ],
            check=True,
            capture_output=capture_output,
            text=True,
        )

        # If we captured output and verbosity is 2, print it
        if capture_output and verbosity >= 2:
            print(result.stdout)

        transcript_file = mp4_path.with_suffix(".txt")
        return transcript_file.read_text() if transcript_file.exists() else ""
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Whisper on {mp4_path.name}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error processing MP4 {mp4_path.name}: {e}")
        return ""


# ============ TEXT COMBINATION ============


def process_single_document(file: Path, verbosity: int = 1) -> str:
    """Process a single document and return its extracted text."""
    ext = file.suffix.lower()
    if ext == ".pdf":
        logging.info(f"Processing PDF: {file.name}")
        return f"PDF: {file.name}\n---\n{extract_text_from_pdf(file)}\n\n"
    elif ext == ".pptx":
        logging.info(f"Processing PPTX: {file.name}")
        return f"PPTX: {file.name}\n---\n{extract_text_from_pptx(file)}\n\n"
    elif ext == ".mp4":
        logging.info(f"Processing MP4: {file.name}")
        return f"MP4: {file.name}\n---\n{transcribe_audio_with_whisper(file, verbosity)}\n\n"
    return ""


def query_openai_for_topics(text: str, verbosity: int = 1) -> str:
    logging.info("Asking OpenAI for structured topics...")

    # Get the path to the system prompt file relative to the script
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    system_prompt_path = script_dir / "system_prompt.txt"

    logging.debug(f"Looking for system prompt at: {system_prompt_path}")

    if not system_prompt_path.exists():
        logging.error(f"System prompt file not found at {system_prompt_path}")
        logging.error(f"Current working directory: {Path.cwd()}")
        raise FileNotFoundError(f"System prompt file not found at {system_prompt_path}")

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    # Log system prompt if verbosity is high
    if verbosity >= 2:
        logging.debug(f"System prompt:\n{system_prompt}")

    # Warn if text is long but don't truncate
    if len(text) > 5000:
        logging.warning(
            f"Input text length is {len(text)} characters. This may take longer to process."
        )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Given the lecture below, extract a concise list of important concepts to turn into individual note titles:\n\n{text}",
            },
        ],
        temperature=0.4,
    )

    # Log raw response if verbosity is high
    if verbosity >= 2:
        logging.debug(f"Raw OpenAI response:\n{response}")

    topics = response.choices[0].message.content
    logging.debug(f"Topics response:\n{topics}")
    return topics


def calculate_token_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate the cost of an API call based on token usage."""
    # Current OpenAI pricing (as of 2024)
    PRICING = {
        "gpt-4": {
            "input": 0.03 / 1000,  # $0.03 per 1K tokens
            "output": 0.06 / 1000,  # $0.06 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.001 / 1000,  # $0.001 per 1K tokens
            "output": 0.002 / 1000,  # $0.002 per 1K tokens
        },
    }

    if model not in PRICING:
        return 0.0

    input_cost = prompt_tokens * PRICING[model]["input"]
    output_cost = completion_tokens * PRICING[model]["output"]
    return input_cost + output_cost


def summarize_with_gpt35(text: str, verbosity: int = 1) -> tuple[str, float]:
    """Summarize text using GPT-3.5-turbo to reduce token usage."""
    logging.info("Summarizing with GPT-3.5-turbo...")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise summaries of educational content. Focus on key concepts and main points.",
            },
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following content, focusing on key concepts and main points:\n\n{text}",
            },
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    # Calculate cost
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost = calculate_token_cost("gpt-3.5-turbo", prompt_tokens, completion_tokens)

    if verbosity >= 1:
        logging.info(
            f"GPT-3.5-turbo usage: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens"
        )
        logging.info(f"Cost: ${cost:.4f}")

    summary = response.choices[0].message.content
    if verbosity >= 2:
        logging.debug(f"Summary:\n{summary}")
    return summary, cost


def chunk_text_with_overlap(
    text: str, max_tokens: int = 4000, overlap_paragraphs: int = 5
) -> list[str]:
    """Split text into chunks with optimal overlap to maintain context."""
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0
    overlap_buffer = []

    for para in paragraphs:
        para_length = len(para)

        # If adding this paragraph would exceed the limit
        if current_length + para_length > max_chars:
            # Save current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            # Start new chunk with optimal overlap
            current_chunk = overlap_buffer + [para]
            current_length = (
                sum(len(p) for p in current_chunk) + (len(current_chunk) - 1) * 2
            )

            # Update overlap buffer with optimal number of paragraphs
            overlap_buffer = current_chunk[-overlap_paragraphs:]
        else:
            current_chunk.append(para)
            current_length += para_length + 2
            if len(current_chunk) > overlap_paragraphs:
                overlap_buffer = current_chunk[-overlap_paragraphs:]

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_document_chunks(text: str, verbosity: int = 1) -> tuple[str, float]:
    """Process a document's text in chunks with summarization."""
    total_cost = 0.0

    # First pass: Split into chunks with optimal overlap
    chunks = chunk_text_with_overlap(text)
    logging.info(f"Split document into {len(chunks)} chunks for processing")

    # Second pass: Summarize each chunk with GPT-3.5-turbo
    summaries = []
    for i, chunk in enumerate(chunks, 1):
        logging.info(f"Summarizing chunk {i}/{len(chunks)} with GPT-3.5-turbo")
        try:
            summary, cost = summarize_with_gpt35(chunk, verbosity)
            total_cost += cost
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error summarizing chunk {i}: {e}")
            continue

    # Combine summaries and get final topics with GPT-4
    combined_summary = "\n\n".join(summaries)
    logging.info("Generating final topics with GPT-4")
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates structured notes from educational content.",
                },
                {
                    "role": "user",
                    "content": f"Given the following summaries, create a structured list of topics and their content:\n\n{combined_summary}",
                },
            ],
            temperature=0.4,
        )

        # Calculate GPT-4 cost
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        gpt4_cost = calculate_token_cost("gpt-4", prompt_tokens, completion_tokens)
        total_cost += gpt4_cost

        if verbosity >= 1:
            logging.info(
                f"GPT-4 usage: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens"
            )
            logging.info(f"Cost: ${gpt4_cost:.4f}")
            logging.info(f"Total cost for document: ${total_cost:.4f}")

        return response.choices[0].message.content, total_cost
    except Exception as e:
        logging.error(f"Error generating final topics: {e}")
        return "", total_cost


def combine_inputs_to_text(resource_folder: Path, verbosity: int = 1) -> str:
    """Process each document separately and combine the results."""
    combined_text = ""
    total_cost = 0.0

    for file in resource_folder.iterdir():
        if file.suffix.lower() in [".pdf", ".pptx", ".mp4"]:
            try:
                # Process each document separately
                doc_text = process_single_document(file, verbosity)
                if doc_text:
                    # Process the document in chunks with summarization
                    topics, cost = process_document_chunks(doc_text, verbosity)
                    total_cost += cost
                    combined_text += f"Document: {file.name}\n"
                    combined_text += topics + "\n\n"
            except Exception as e:
                logging.error(f"Error processing {file.name}: {e}")
                continue

    if verbosity >= 1:
        logging.info(f"Total processing cost: ${total_cost:.4f}")

    return combined_text


# ============ OPENAI PROCESSING ============


# ============ NOTE CREATION ============


def find_obsidian_vault_root(path: Path) -> Path:
    """Find the Obsidian vault root directory by looking for .obsidian/ directory."""
    current = path
    while current != current.parent:  # Stop at root directory
        if (current / ".obsidian").exists():
            return current
        current = current.parent
    return None


def generate_tag_path(vault_root: Path, current_path: Path) -> str:
    """Generate the tag path from vault root to current directory."""
    if not vault_root:
        return ""
    # Get relative path from vault root
    rel_path = current_path.relative_to(vault_root)
    # Convert to string and remove whitespace
    tag_path = str(rel_path).replace(" ", "")
    return tag_path


def create_obsidian_notes(
    topics: str, output_dir: Path, interactive: bool = False
) -> None:
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Output directory: {output_dir}")

    # Find Obsidian vault root and generate tag path
    vault_root = find_obsidian_vault_root(output_dir)
    tag_path = generate_tag_path(vault_root, output_dir)
    base_tags = ["todo"]
    if tag_path:
        base_tags.append(tag_path)
    tags_str = ", ".join(base_tags)
    logging.debug(f"Tags: {tags_str}")

    # Split the topics string into individual files
    files_combined = topics.split("---")
    logging.debug(f"Number of topic sections: {len(files_combined)}")
    files = []
    i = 0
    while i < len(files_combined) - 1:
        file_name = files_combined[i]
        if not file_name:
            i += 1
            continue
        file_content = files_combined[i + 1]
        if not file_content:
            i += 1
            continue

        file_name = "".join(c for c in file_name if c.isalnum() or c in "._-")
        # Check if filename has an extension
        if "." in file_name:
            # Replace existing extension with .md
            file_name = file_name.rsplit(".", 1)[0] + ".md"
        elif not file_name.endswith(".md"):
            # Add .md extension if no extension exists
            file_name += ".md"
        files.append((file_name, file_content))
        logging.debug(f"Prepared file: {file_name}")
        i += 2

    logging.debug(f"Total files to create: {len(files)}")
    for filename, content in files:
        if interactive:
            print(f"\nFile to be created: {filename}")
            print(f"Content preview:\n{content}...")  # Show first 200 chars
            while True:
                response = input("Create this file? (y/n/a): ").lower()
                if response == "y":
                    break
                elif response == "n":
                    logging.info(f"Skipping file: {filename}")
                    continue
                elif response == "a":
                    logging.info("Aborting all file creation")
                    return
                else:
                    print("Invalid input. Please enter y, n, or a.")
        else:
            logging.info(f"Creating file: {filename}")

        # Create the file with YAML frontmatter
        filepath = output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Write YAML frontmatter
                f.write("---\n")
                f.write(f"tags: {tags_str}\n")
                f.write("---\n\n")
                # Write content
                f.write(content)
            logging.info(f"Created: {filepath.name}")
        except Exception as e:
            logging.error(f"Error creating file {filepath.name}: {e}")
            logging.error(f"Error details: {str(e)}")


# ============ MAIN ============


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured notes from learning resources."
    )
    parser.add_argument("lecture_dir", help="Path to the lecture directory")
    parser.add_argument(
        "-v0",
        action="store_const",
        const=0,
        dest="verbosity",
        help="Silent mode (errors only)",
    )
    parser.add_argument(
        "-v1",
        action="store_const",
        const=1,
        dest="verbosity",
        help="Basic progress output (default)",
    )
    parser.add_argument(
        "-v2",
        action="store_const",
        const=2,
        dest="verbosity",
        help="Detailed output including system prompt and raw responses",
    )
    parser.add_argument("-l", "--log", help="Write output to specified log file")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode for file creation",
    )

    args = parser.parse_args()

    # Set default verbosity if not specified
    if args.verbosity is None:
        args.verbosity = 1

    # Set up logging
    setup_logging(args.verbosity, args.log)

    # Check dependencies first
    check_dependencies()

    # Handle the lecture directory path
    try:
        # First expand any ~ or environment variables
        expanded_path = os.path.expandvars(os.path.expanduser(args.lecture_dir))
        # Then resolve to absolute path
        root_path = Path(expanded_path).resolve()
        resource_path = root_path / "Learning Resources"
    except Exception as e:
        logging.error(f"Error processing path '{args.lecture_dir}': {e}")
        sys.exit(1)

    if not resource_path.exists():
        logging.error(f"{resource_path} does not exist.")
        sys.exit(1)

    logging.info(f"Extracting data from: {resource_path}")
    all_text = combine_inputs_to_text(resource_path, args.verbosity)

    # Write extracted text to a file
    output_file = root_path / "extracted_text.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        logging.info(f"Extracted text written to: {output_file}")
    except Exception as e:
        logging.error(f"Error writing extracted text: {e}")
        sys.exit(1)

    logging.info(f"Creating notes in: {root_path}")
    create_obsidian_notes(all_text, root_path, args.interactive)

    logging.info("Done!")


if __name__ == "__main__":
    main()
