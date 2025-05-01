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
	@echo "Installing gen-notes utilities..."
	@if grep -q "alias gen-notes=" ~/.zshrc; then \
		echo "gen-notes alias already exists, skipping..."; \
	else \
		echo 'alias gen-notes="source $(VENV_DIR)/bin/activate && $(PYTHON) \"$(REPO_DIR)/gen_notes.py\" \"\$$@\""' >> ~/.zshrc; \
	fi
	@if grep -q "gen-notes-concur()" ~/.zshrc; then \
		echo "gen-notes-concur function already exists, skipping..."; \
	else \
		echo 'gen-notes-concur() { printf "%s\n" "$$@" | xargs -n1 -P$$(sysctl -n hw.ncpu) -I{} gen-notes {} }' >> ~/.zshrc; \
	fi
	@echo "Reloading shell configuration..."
	@source ~/.zshrc
	@echo "Installation complete! You can now use:"
	@echo "  gen-notes <lecture_dir> [options]"
	@echo "  gen-notes-concur <lecture_dir1> <lecture_dir2> ..."

uninstall:
	@echo "Removing gen-notes utilities..."
	@sed -i '' '/alias gen-notes=/d' ~/.zshrc
	@sed -i '' '/gen-notes-concur()/d' ~/.zshrc
	@echo "Reloading shell configuration..."
	@source ~/.zshrc
	@echo "Utilities removed"

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Virtual environment removed" 