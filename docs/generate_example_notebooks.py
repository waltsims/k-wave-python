#!/usr/bin/env python3
"""Convert examples/*.py (jupytext percent format) to notebooks/*.ipynb.

Run this before the Sphinx build so that nbsphinx can render them.

Usage:
    python docs/generate_example_notebooks.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
OUTPUT_DIR = REPO_ROOT / "notebooks"

_INSTALL_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": "!pip install k-wave-python",
}


def _promote_docstring_to_markdown(nb_path: Path) -> None:
    """If the first cell is a code cell whose only content is a docstring,
    convert it to a markdown cell with the first line as an H1 title.
    Then insert a pip install cell for Colab users."""
    with open(nb_path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells:
        return

    first = cells[0]
    if first["cell_type"] != "code":
        return

    source = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
    stripped = source.strip()

    # Detect triple-quoted docstring: starts and ends with """ (or '''),
    # with no other triple-quotes in between.
    for quote in ('"""', "'''"):
        if stripped.startswith(quote) and stripped.endswith(quote) and len(stripped) > 6:
            inner = stripped[3:-3]
            if quote not in inner:
                body = inner.strip()
                break
    else:
        return

    # Split into title (first line) and description (rest)
    lines = body.split("\n", 1)
    title = lines[0].strip()
    description = lines[1].strip() if len(lines) > 1 else ""

    md_source = f"# {title}"
    if description:
        md_source += f"\n\n{description}"

    first["cell_type"] = "markdown"
    first["source"] = md_source
    first.pop("execution_count", None)
    first.pop("outputs", None)

    # Insert pip install cell after title, before imports
    cells.insert(1, _INSTALL_CELL.copy())

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
        f.write("\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    py_files = sorted(EXAMPLES_DIR.glob("*.py"))
    if not py_files:
        print("No example .py files found.", file=sys.stderr)
        sys.exit(1)

    for py_file in py_files:
        out_file = OUTPUT_DIR / py_file.with_suffix(".ipynb").name
        print(f"  {py_file.name} -> {out_file.relative_to(REPO_ROOT)}")
        subprocess.check_call(
            [sys.executable, "-m", "jupytext", "--to", "notebook", str(py_file), "-o", str(out_file)],
        )
        _promote_docstring_to_markdown(out_file)

    print(f"Converted {len(py_files)} examples to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
