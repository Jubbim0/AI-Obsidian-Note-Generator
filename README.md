# AI Note Workflow

A Python script that automatically generates structured notes from various learning resources (PDFs, PowerPoint presentations, and video lectures) using OpenAI's GPT-4.

## Features

- Extracts text from PDF files
- Extracts text from PowerPoint presentations
- Transcribes audio from video files using Whisper
- Processes content using OpenAI's GPT-4
- Creates structured markdown notes
- Handles multiple file types in a single run
- Configurable verbosity levels
- Optional logging to file
- Interactive file creation mode

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

# Install the required dependencies
pip install pypdf python-pptx python-dotenv openai

# Install Whisper for audio transcription
pip install git+https://github.com/openai/whisper.git

# Deactivate the virtual environment when done
deactivate
```

4. Add the command-line utilities to your shell:
```bash
# Add these lines to your ~/.zshrc file
echo 'alias gen-notes="source ~/.gen-notes-venv/bin/activate && python3 \"'$(pwd)'/gen_notes.py\" \"\$@\""' >> ~/.zshrc
echo 'gen-notes-concur() { printf "%s\n" "$@" | xargs -n1 -P$(sysctl -n hw.ncpu) -I{} gen-notes {} }' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

5. Verify the installation:
```bash
# Test the gen-notes command
gen-notes --help

# Test the concurrent version
gen-notes-concur --help
```

The commands will now be available in your shell:
- `gen-notes <lecture_dir> [options]` for single directory processing
- `gen-notes-concur <lecture_dir1> <lecture_dir2> ...` for concurrent processing of multiple directories

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
Lecture_Name/
└── Learning Resources/
    ├── lecture.pdf
    ├── slides.pptx
    └── video.mp4
```

2. Run the script with optional flags:
```bash
python gen_notes.py /path/to/Lecture_Name [options]
```

### Command Line Options

- `-v0`: Silent mode - only outputs errors
- `-v1`: Basic progress output (default)
- `-v2`: Detailed output including system prompt and raw responses
- `-l <filename>`: Write all output to specified log file
- `-i`: Interactive mode for file creation

### Interactive Mode

When using the `-i` flag, the script will prompt you before creating each file:
- `y`: Create the file
- `n`: Skip this file
- `a`: Abort all remaining file creation

### Examples

Basic usage:
```bash
python gen_notes.py /path/to/Lecture_Name
```

Silent mode with logging:
```bash
python gen_notes.py /path/to/Lecture_Name -v0 -l output.log
```

Interactive mode with detailed output:
```bash
python gen_notes.py /path/to/Lecture_Name -v2 -i
```

## Testing

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run the tests:
```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=gen_notes tests/

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
- PowerPoint (.pptx)
- Video (.mp4) - requires Whisper for transcription

## Output

The script generates:
1. An `extracted_text.txt` file containing all extracted content
2. Individual markdown files for each topic identified by GPT-4
3. Optional log file if `-l` flag is used

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
