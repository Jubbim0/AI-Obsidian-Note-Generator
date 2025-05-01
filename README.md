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

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Jubbim0/ai_note_workflow.git
cd ai_note_workflow
```

2. Install the required dependencies:
```bash
pip install pypdf python-pptx python-dotenv openai
```

3. Install Whisper for audio transcription:
```bash
pip install git+https://github.com/openai/whisper.git
```

## Environment Setup

1. Create a `.env` file in the project root directory
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys).

## Usage

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
