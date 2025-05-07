"""Setup file for the AI Note Workflow package."""

from setuptools import setup, find_packages

setup(
    name="ai_note_workflow",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-docx",
        "PyPDF2",
        "python-pptx",
        "openpyxl",
        "pandas",
        "requests",
        "tqdm",
        "python-dotenv",
        "numpy",
    ],
    extras_require={
        "audio": [
            "torch",
            "whisper",
        ],
    },
    entry_points={
        "console_scripts": [
            "gen-notes=src:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Note Workflow - Generate structured notes from various learning resources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai_note_workflow",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
