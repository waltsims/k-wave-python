# Define the Python interpreter and MATLAB command
PYTHON = python
MATLAB = matlab

# Define the directories and files
EXAMPLES_DIR = examples
TESTS_DIR = tests
MATLAB_SCRIPT = tests/matlab_test_data_collectors/run_all_collectors.m
KWAVE_MATLAB_PATH = $(abspath ../k-wave) # Get absolute path of k-wave directory

# Define the artifact directory for collected test values
COLLECTED_VALUES_DIR = $(abspath tests/matlab_test_data_collectors/python_testers/collectedValues)

# Platform-specific MATLAB detection
# On macOS, automatically find the latest MATLAB installation
ifeq ($(shell uname), Darwin)
    # Find the latest MATLAB version installed in /Applications
    # Sort in reverse order and take the first (latest) version
    MATLAB_PATH := $(shell ls -d /Applications/MATLAB_R* 2>/dev/null | sort -r | head -n 1)
    ifneq ($(MATLAB_PATH),)
        MATLAB := $(MATLAB_PATH)/bin/matlab
    endif
endif

# Default target - runs all examples and tests
all: run-examples test

# Target to run all examples with non-interactive backend
run-examples:
	@echo "Running all examples..."
	@MPLBACKEND=Agg $(PYTHON) run_examples.py

# Target to run pytest, which depends on running the MATLAB script first
test: $(COLLECTED_VALUES_DIR)
	@echo "Running pytest..."
	@pytest $(TESTS_DIR)

# Target to run the MATLAB script and create the artifact directory
# Includes fallback behavior if MATLAB is not available
$(COLLECTED_VALUES_DIR): 
	@echo "Running MATLAB script to collect values..."; \
	if [ -f "$(MATLAB)" ]; then \
		$(MATLAB) -batch "run('$(MATLAB_SCRIPT)');"; \
	else \
		echo "MATLAB not found. Skipping MATLAB data collection."; \
		echo "Some tests may fail if they depend on MATLAB data."; \
		echo "To run all tests, you need to:"; \
		echo "1. Install MATLAB from https://www.mathworks.com/products/matlab.html"; \
		echo "2. Make sure MATLAB is in your PATH or installed in /Applications/MATLAB_R* on macOS"; \
		echo "3. Run 'make clean' to clean any partial test data"; \
		echo "4. Run 'make test' again"; \
		echo ""; \
		echo "For now, continuing with available tests..."; \
		mkdir -p $(COLLECTED_VALUES_DIR); \
	fi

# Clean target - removes all generated files and caches
clean: clean-python clean-collected_values

# Clean Python-specific files
clean-python:
	@echo "Cleaning Python cache files..."
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -delete

# Clean collected test values
clean-collected_values:
	@echo "Cleaning collected values directory..."
	@rm -rf $(COLLECTED_VALUES_DIR)

.PHONY: all run-examples run-tests clean-python clean-collected_values clean
