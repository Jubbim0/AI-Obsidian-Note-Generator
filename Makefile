.PHONY: install uninstall check-deps clean help

# Variables
VENV_DIR := $(HOME)/.gen-notes-venv
REPO_DIR := $(shell pwd)
PYTHON := python3
PIP := pip3

help:
	@echo "Available targets:"
	@echo "  install     - Install gen-notes and gen-notes-concur utilities"
	@echo "  uninstall   - Remove gen-notes and gen-notes-concur utilities"
	@echo "  check-deps  - Check if all dependencies are installed"
	@echo "  clean       - Remove virtual environment"
	@echo "  help        - Show this help message"

check-deps:
	@echo "Checking dependencies..."
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Error: python3 is not installed"; exit 1; }
	@command -v $(PIP) >/dev/null 2>&1 || { echo "Error: pip3 is not installed"; exit 1; }
	@echo "All dependencies are installed"

$(VENV_DIR):
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing dependencies..."
	$(VENV_DIR)/bin/pip install pypdf python-pptx python-dotenv openai

install: check-deps $(VENV_DIR)
	@echo "Installing command-line utilities..."
	@# Check if aliases already exist
	@if grep -q "alias gen-notes=" ~/.zshrc; then \
		echo "Alias 'gen-notes' already exists in ~/.zshrc"; \
		exit 1; \
	fi
	@if grep -q "gen-notes-concur()" ~/.zshrc; then \
		echo "Function 'gen-notes-concur' already exists in ~/.zshrc"; \
		exit 1; \
	fi
	@# Add aliases with proper quoting for paths with spaces
	@echo 'alias gen-notes="source ~/.gen-notes-venv/bin/activate && python3 \"'$(REPO_DIR)'/gen_notes.py\" \"\$$@\""' >> ~/.zshrc
	@echo 'gen-notes-concur() { for dir in "$$@"; do gen-notes "$$dir" & done; wait; }' >> ~/.zshrc
	@echo "Installation complete! Please restart your shell or run 'source ~/.zshrc'"

uninstall:
	@echo "Removing gen-notes utilities..."
	@if [ -f ~/.zshrc ]; then \
		sed -i '' '/alias gen-notes=/d' ~/.zshrc; \
		sed -i '' '/gen-notes-concur()/d' ~/.zshrc; \
		echo "Aliases removed from ~/.zshrc"; \
		echo "Please restart your shell or run 'source ~/.zshrc' to apply changes"; \
	else \
		echo "Warning: ~/.zshrc not found or not a regular file"; \
	fi
	@echo "Utilities removed"

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Virtual environment removed" 