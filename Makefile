# Define the Python interpreter and MATLAB command
PYTHON = python
MATLAB = matlab

# Define the directories and files
EXAMPLES_DIR = examples
TESTS_DIR = tests
MATLAB_SCRIPT = tests/matlab_test_data_collectors/run_all_collectors.m
KWAVE_MATLAB_PATH = $(abspath ../k-wave) # Get absolute path of k-wave directory

# Define the artifact directory
COLLECTED_VALUES_DIR = $(abspath tests/matlab_test_data_collectors/python_testers/collectedValues)

# Default target
all: run-examples test

# Target to run all examples
run-examples:
	@echo "Running all examples..."
	@MPLBACKEND=Agg $(PYTHON) run_examples.py

# Target to run pytest, which depends on running the MATLAB script first
test: $(COLLECTED_VALUES_DIR)
	@echo "Running pytest..."
	@pytest $(TESTS_DIR)

# Target to run the MATLAB script and create the artifact directory
$(COLLECTED_VALUES_DIR): 
	@echo "Running MATLAB script to collect values..."; \
	$(MATLAB) -batch "run('$(MATLAB_SCRIPT)');"; \
# Clean target (optional) - cleans Python caches and collected values
clean: clean-python clean-collected_values

clean-python:
	@echo "Cleaning Python cache files..."
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -delete

# Clean collected values directory
clean-collected_values:
	@echo "Cleaning collected values directory..."
	@rm -rf $(COLLECTED_VALUES_DIR)

.PHONY: all run-examples run-tests clean-python clean-collected_values clean
