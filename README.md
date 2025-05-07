# AI Note Workflow

A Python package that automatically generates structured notes from various learning resources (PDFs, PowerPoint presentations, and video lectures) using OpenAI's GPT-4.

## Features

- Extracts text from PDF files
- Extracts text from PowerPoint presentations (.pptx and .ppt)
- Transcribes audio from video files using Whisper
- Processes content using OpenAI's GPT-4
- Creates structured markdown notes
- Handles multiple file types in a single run
- Configurable verbosity levels
- Optional logging to file
- Interactive file creation mode
- Cost-optimized processing with detailed cost tracking
- Automatic topic deduplication across multiple resources
- Context-aware topic generation using existing topics

# Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/Jubbim0/ai_note_workflow.git
cd ai_note_workflow
```

2. Create a `.env` file in the project root directory, and Add your OpenAI API key (You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys).):
```
OPENAI_API_KEY=your-api-key-here
```

3. Create and configure a Python virtual environment:
```bash
# Create a virtual environment in your home directory
python3 -m venv ~/.gen-notes-venv

# Activate the virtual environment
source ~/.gen-notes-venv/bin/activate

# Install the package in development mode
pip install -e .

# Deactivate the virtual environment when done
deactivate
```

4. Add the command-line utilities to your shell:
```bash
# Add these lines to your ~/.zshrc file
echo 'alias gen-notes="source ~/.gen-notes-venv/bin/activate && python3 -m src \"\$@\""' >> ~/.zshrc
echo 'gen-notes-concur() { for dir in "$@"; do gen-notes "$dir" & done; wait; }' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

5. Verify the installation:
```bash
# Test the gen-notes command with a path containing spaces
gen-notes "/path/to/My Lecture Notes"

# Test the concurrent version with multiple paths
gen-notes-concur "/path/to/Lecture 1" "/path/to/Lecture 2"
```

The commands will now be available in your shell:
- `gen-notes "<lecture_dir>" [options]` for single directory processing
- `gen-notes-concur "<lecture_dir1>" "<lecture_dir2>" ...` for concurrent processing of multiple directories

To remove the utilities:
```bash
# Remove the aliases from your ~/.zshrc
sed -i '' '/alias gen-notes=/d' ~/.zshrc
sed -i '' '/gen-notes-concur()/d' ~/.zshrc

# Remove the virtual environment
rm -rf ~/.gen-notes-venv

# Reload your shell configuration
source ~/.zshrc
```

# Quick Installation
1. Clone this repository:
```bash
git clone https://github.com/Jubbim0/ai_note_workflow.git
cd ai_note_workflow
```

2. Create a `.env` file in the project root directory, and Add your OpenAI API key (You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys).):
```
OPENAI_API_KEY=your-api-key-here
```

After cloning the repository, you can install the command-line utilities using the provided Makefile:
```bash
# Install the utilities
make install

# To remove the utilities
make uninstall

# To clean up (remove virtual environment)
make clean
```

Running `make install` will:
1. Check for required dependencies
2. Create a Python virtual environment
3. Install all required packages
4. Add the `gen-notes` and `gen-notes-concur` commands to your shell
5. Configure everything to work with paths containing spaces and special characters

After installation, you can use:
- `gen-notes <lecture_dir> [options]` for single directory processing
- `gen-notes-concur <lecture_dir1> <lecture_dir2> ...` for concurrent processing of multiple directories

# Usage

1. Create a directory structure for your lecture:
```
Lecture Name/  # Note: Spaces in directory names are supported
└── Learning Resources/
    ├── lecture.pdf
    ├── slides.pptx
    └── video.mp4
```

2. Run the script with optional flags:
```bash
# Basic usage
python -m src "/path/to/Lecture Name"

# Using quotes for paths with spaces
python -m src "/path/to/My Lecture Notes"

# Using environment variables
python -m src "$HOME/Documents/My Lectures"

# Using relative paths
python -m src "./My Lecture Notes"
```

### Path Handling
- The script supports paths with spaces when properly quoted
- Environment variables (like `$HOME`) are expanded
- Tilde expansion (`~`) is supported
- Both absolute and relative paths work
- Paths are automatically resolved to their absolute form

### Command Line Options

- `-v0`: Silent mode - only outputs errors
- `-v1`: Basic progress output (default)
  - Shows chunk processing progress
  - Displays cost information
  - Reports duplicate topic merging
- `-v2`: Detailed output including:
  - System prompt and raw responses
  - Cost tracking details
  - Chunk sizes and overlap information
  - Topic deduplication details
  - Content length statistics

### Cost Optimization
The script uses a cost-optimized approach:
1. Each document is processed in a single GPT-4 call
2. Topics are merged automatically to reduce redundancy
3. Existing topics are considered when generating new topics
4. Detailed cost tracking shows:
   - Token usage per API call
   - Cost per document
   - Total processing cost

### Content Processing
- Each document is processed independently
- Topics are extracted and merged automatically
- Error handling ensures partial success
- Duplicate topics from different resources are automatically merged
- Content from all sources is preserved in merged topics
- Existing topics are considered for better topic relationships

### Examples

Basic usage:
```bash
python -m src "/path/to/Lecture Name"
```

Detailed output with cost tracking:
```bash
python -m src "/path/to/Lecture Name" -v2
```

Interactive mode with detailed output:
```bash
python -m src "/path/to/Lecture Name" -v2 -i
```

## Testing

1. Install test dependencies:
```bash
pip install -e ".[test]"
```

2. Run the tests:
```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=src tests/

# Run specific test
pytest tests/test_gen_notes.py::test_name

# Run with verbose output
pytest -v tests/
```

The test suite includes:
- PDF text extraction tests
- PowerPoint text extraction tests
- Text combination tests
- Note creation tests
- Logging setup tests
- Error handling tests

## Supported File Types

- PDF (.pdf)
- PowerPoint (.pptx, .ppt)
- Video (.mp4) - requires Whisper for transcription

## Output

The script generates:
1. Individual markdown files for each topic identified by GPT-4
2. Optional log file if `-l` flag is used

## Error Handling

The script includes comprehensive error handling for:
- Missing dependencies
- File processing errors
- API key issues
- File system operations

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls
- Sufficient disk space for temporary files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive free software license that allows anyone to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the condition that the original copyright notice and permission notice be included in all copies or substantial portions of the software.
