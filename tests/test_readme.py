import os
import re
import subprocess
import sys
from tempfile import mkdtemp

import pytest
import requests


def test_readme():
    # Check if there is internet connectivity
    try:
        requests.get("https://google.com")
    except requests.exceptions.ConnectionError:
        pytest.skip("No internet connectivity")

    # Skip the test if the operating system is MacOS
    if sys.platform.startswith('darwin'):
        pytest.skip("This test cannot be run on MacOS")

    # Read the getting started section from the getting_started.md file
    with open('../README.md', 'r') as f:
        readme = f.read()

    tempdir = mkdtemp()
    os.chdir(tempdir)
    # Use a regular expression to find code blocks in the getting started section
    code_blocks = re.findall(r'```bash(.*?)```', readme, re.DOTALL)

    for block in code_blocks:
        instruction = block
        result = subprocess.run(['bash', '-c', instruction])
        assert result.returncode == 0, f"instruction failed: {instruction}"
    pass
