import os
from pathlib import Path
import subprocess


def run_python_files(directory):
    # Recursively walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if it's a Python file
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Running: {file_path}")
                try:
                    # Use subprocess to run the Python file
                    result = subprocess.run(["python", file_path], capture_output=True, text=True)
                    print(f"Output:\n{result.stdout}")
                    if result.stderr:
                        print(f"Errors:\n{result.stderr}")
                except Exception as e:
                    print(f"Failed to run {file_path}: {e}")


if __name__ == "__main__":
    directory = Path("examples/")
    run_python_files(directory)
