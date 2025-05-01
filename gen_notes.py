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


def combine_inputs_to_text(resource_folder: Path, verbosity: int = 1) -> str:
    combined_text = ""
    for file in resource_folder.iterdir():
        ext = file.suffix.lower()
        if ext == ".pdf":
            logging.info(f"Processing PDF: {file.name}")
            combined_text += f"PDF: {file.name}\n---\n"
            combined_text += extract_text_from_pdf(file) + "\n\n"
            combined_text += "---" + "\n\n"
        elif ext == ".pptx":
            logging.info(f"Processing PPTX: {file.name}")
            combined_text += f"PPTX: {file.name}\n---\n"
            combined_text += extract_text_from_pptx(file) + "\n\n"
            combined_text += "---" + "\n\n"
        elif ext == ".mp4":
            logging.info(f"Processing MP4: {file.name}")
            combined_text += f"MP4: {file.name}\n---\n"
            combined_text += transcribe_audio_with_whisper(file, verbosity) + "\n\n"
            combined_text += "---" + "\n\n"
    return combined_text


# ============ OPENAI PROCESSING ============


def query_openai_for_topics(text: str, verbosity: int = 1) -> str:
    logging.info("Asking OpenAI for structured topics...")

    # Get the path to the system prompt file relative to the script
    script_dir = Path(__file__).parent
    system_prompt_path = script_dir / "system_prompt.txt"

    if not system_prompt_path.exists():
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

    return response.choices[0].message.content


# ============ NOTE CREATION ============


def create_obsidian_notes(
    topics: str, output_dir: Path, interactive: bool = False
) -> None:
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the topics string into individual files
    files_combined = topics.split("---")
    logging.debug(files_combined)
    files = []
    i = 0
    while i < len(files_combined) - 1:
        file_name = files_combined[i]
        if not file_name:
            continue
        file_content = files_combined[i + 1]
        if not file_content:
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
        i += 2

    for filename, content in files:
        if interactive:
            print(f"\nFile to be created: {filename}")
            print(f"Content preview:\n{content[:200]}...")  # Show first 200 chars
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

        # Create the file
        filepath = output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Created: {filepath.name}")
        except Exception as e:
            logging.error(f"Error creating file {filepath.name}: {e}")


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

    root_path = Path(args.lecture_dir).expanduser().resolve()
    resource_path = root_path / "Learning Resources"

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

    logging.info("Sending data to OpenAI...")
    try:
        topics = query_openai_for_topics(all_text, args.verbosity)
    except Exception as e:
        logging.error(f"Error processing with OpenAI: {e}")
        sys.exit(1)

    logging.info(f"Creating notes in: {root_path}")
    create_obsidian_notes(topics, root_path, args.interactive)

    logging.info("Done!")


if __name__ == "__main__":
    main()
