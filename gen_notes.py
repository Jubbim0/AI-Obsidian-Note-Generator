import sys
import subprocess
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

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

# ============ FILE HANDLERS ============


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import PyPDF2
        import pptx
    except ImportError as e:
        print(f"Error: Required dependency not installed: {e}")
        print("Please install required packages: pip install PyPDF2 python-pptx")
        sys.exit(1)


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"Error processing PDF {pdf_path.name}: {e}")
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
        print(f"Error processing PPTX {pptx_path.name}: {e}")
        return ""


def transcribe_audio_with_whisper(mp4_path: Path) -> str:
    try:
        print(f"Transcribing {mp4_path.name} using Whisper...")
        subprocess.run(
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
        )
        transcript_file = mp4_path.with_suffix(".txt")
        return transcript_file.read_text() if transcript_file.exists() else ""
    except subprocess.CalledProcessError as e:
        print(f"Error running Whisper on {mp4_path.name}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing MP4 {mp4_path.name}: {e}")
        return ""


# ============ TEXT COMBINATION ============


def combine_inputs_to_text(resource_folder: Path) -> str:
    combined_text = ""
    for file in resource_folder.iterdir():
        ext = file.suffix.lower()
        if ext == ".pdf":
            print(f"Processing PDF: {file.name}")
            combined_text += f"PDF: {file.name}\n---\n"
            combined_text += extract_text_from_pdf(file) + "\n\n"
            combined_text += "---" + "\n\n"
        elif ext == ".pptx":
            print(f"Processing PPTX: {file.name}")
            combined_text += f"PPTX: {file.name}\n---\n"
            combined_text += extract_text_from_pptx(file) + "\n\n"
            combined_text += "---" + "\n\n"
        elif ext == ".mp4":
            print(f"Processing MP4: {file.name}")
            combined_text += f"MP4: {file.name}\n---\n"
            combined_text += transcribe_audio_with_whisper(file) + "\n\n"
            combined_text += "---" + "\n\n"
    return combined_text


# ============ OPENAI PROCESSING ============


def query_openai_for_topics(text: str) -> str:
    print("Asking OpenAI for structured topics...")

    # Get the path to the system prompt file relative to the script
    script_dir = Path(__file__).parent
    system_prompt_path = script_dir / "system_prompt.txt"

    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at {system_prompt_path}")

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    # Warn if text is long but don't truncate
    if len(text) > 5000:
        print(
            f"Warning: Input text length is {len(text)} characters. This may take longer to process."
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
    return response.choices[0].message.content


# ============ NOTE CREATION ============


def create_obsidian_notes(topics: str, output_dir: Path) -> None:
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the topics string into individual files
    files_combined = topics.split("---")
    print(files_combined)
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
        # Create the file
        filepath = output_dir / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Created: {filepath.name}")
        except Exception as e:
            print(f"Error creating file {filepath.name}: {e}")


# ============ MAIN ============


def main(lecture_dir_path: str):
    # Check dependencies first
    check_dependencies()

    root_path = Path(lecture_dir_path).expanduser().resolve()
    resource_path = root_path / "Learning Resources"

    if not resource_path.exists():
        print(f"Error: {resource_path} does not exist.")
        return

    if True:  # testing
        with open("test.txt", "r", encoding="utf-8") as f:
            create_obsidian_notes(f.read(), root_path)
        return

    print(f"🔍 Extracting data from: {resource_path}")
    all_text = combine_inputs_to_text(resource_path)

    # Write extracted text to a file
    output_file = root_path / "extracted_text.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"📝 Extracted text written to: {output_file}")
    except Exception as e:
        print(f"Error writing extracted text: {e}")
        return

    print(f"📤 Sending data to OpenAI...")
    try:
        topics = query_openai_for_topics(all_text)
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return

    print(f"📝 Creating notes in: {root_path}")
    create_obsidian_notes(topics, root_path)

    print("🎉 Done!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gennotes.py /absolute/path/to/Lecture_03_RSA")
    else:
        main(sys.argv[1])
