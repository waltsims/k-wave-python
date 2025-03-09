import os
import sys
import warnings
import importlib
import importlib.util
from pathlib import Path
import tempfile

import pytest

from kwave.utils.deprecation import deprecated


def test_deprecation_warning_at_import_time():
    """Test that deprecation warnings are issued at import time, not call time."""

    # Create a temporary module with a deprecated function
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_file.write(
            b"""
from kwave.utils.deprecation import deprecated

@deprecated("Use new_function() instead", "2.0.0")
def old_function():
    return "I'm deprecated"

def new_function():
    return "I'm the new function"
"""
        )
        temp_module_path = temp_file.name

    try:
        # Get the module name from the file path
        module_name = os.path.basename(temp_module_path).replace(".py", "")

        # Add the directory containing the module to sys.path
        module_dir = os.path.dirname(temp_module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        # First, verify that importing the module triggers the warning
        with warnings.catch_warnings(record=True) as import_warnings:
            warnings.simplefilter("always")  # Ensure all warnings are captured

            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, temp_module_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

        # Check that we got exactly one warning during import
        assert len(import_warnings) == 1, "Expected exactly one warning during import"
        assert issubclass(import_warnings[0].category, DeprecationWarning), "Expected a DeprecationWarning"
        assert "old_function is deprecated" in str(import_warnings[0].message), "Warning message doesn't match expected"

        # Now, verify that calling the function does NOT trigger additional warnings
        with warnings.catch_warnings(record=True) as call_warnings:
            warnings.simplefilter("always")  # Ensure all warnings are captured

            # Call the deprecated function
            result = temp_module.old_function()
            assert result == "I'm deprecated", "Function should still work correctly"

            # Call it again to be sure
            temp_module.old_function()

        # Check that no warnings were issued during function calls
        assert len(call_warnings) == 0, "No warnings should be issued when calling the function"

    finally:
        # Clean up the temporary module file
        if os.path.exists(temp_module_path):
            os.unlink(temp_module_path)

        # Remove the directory from sys.path if we added it
        if module_dir in sys.path:
            sys.path.remove(module_dir)


if __name__ == "__main__":
    test_deprecation_warning_at_import_time()
