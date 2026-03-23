"""Run all example scripts with the Python backend on CPU."""
import os
import subprocess
import sys
from pathlib import Path

SKIP = {"example_utils.py", "__init__.py"}


def run_examples(directory: Path):
    env = {**os.environ, "KWAVE_BACKEND": "python", "KWAVE_DEVICE": "cpu", "MPLBACKEND": "Agg"}
    examples = sorted(p for p in directory.rglob("*.py") if p.name not in SKIP)
    failed = []
    for path in examples:
        print(f"\n{'='*60}\nRunning: {path}\n{'='*60}")
        result = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            if result.stderr:
                print(result.stderr[-500:])
            failed.append(str(path))
        else:
            print("OK")

    print(f"\n{'='*60}")
    print(f"Results: {len(examples) - len(failed)}/{len(examples)} passed")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    run_examples(Path("examples/"))
