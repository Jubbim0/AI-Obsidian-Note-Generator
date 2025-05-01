import pytest
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment."""
    # Create a temporary .env file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("OPENAI_API_KEY=test_api_key\n")
    os.environ["DOTENV_FILE"] = f.name
    yield
    # Cleanup
    os.unlink(f.name)
    if "DOTENV_FILE" in os.environ:
        del os.environ["DOTENV_FILE"]
