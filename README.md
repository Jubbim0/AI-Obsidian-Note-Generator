# AI Note Workflow

A Python script that automatically generates structured notes from various learning resources (PDFs, PowerPoint presentations, and video lectures) using OpenAI's GPT-4.

## Features

- Extracts text from PDF files
- Extracts text from PowerPoint presentations
- Transcribes audio from video files using Whisper
- Processes content using OpenAI's GPT-4
- Creates structured markdown notes
- Handles multiple file types in a single run

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Jubbim0/ai_note_workflow.git
cd ai_note_workflow
```

2. Install the required dependencies:
```bash
pip install PyPDF2 python-pptx python-dotenv openai
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

2. Run the script:
```bash
python gen_notes.py /path/to/Lecture_Name
```

The script will:
1. Extract text from all supported files in the "Learning Resources" directory
2. Process the content using GPT-4
3. Create structured markdown notes in the lecture directory

## Supported File Types

- PDF (.pdf)
- PowerPoint (.pptx)
- Video (.mp4) - requires Whisper for transcription

## Output

The script generates:
1. An `extracted_text.txt` file containing all extracted content
2. Individual markdown files for each topic identified by GPT-4

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

[Add your license here]
