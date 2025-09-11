# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "k-Wave-python"
copyright = "2024, Walter Simson, Farid Yagubbayli"
author = "Walter Simson, Farid Yagubbayli"
version = "0.3.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_toolbox.code",
    "sphinx_copybutton",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_mdinclude",
    "sphinx.ext.extlinks",  # Enables :ghfile: external links
]

# Provide a simple GitHub branch/sha fallback for linking to example files.
# Priority: explicit override via _GITHUB_BRANCH →
#           READTHEDOCS_GIT_IDENTIFIER (commit) →
#           READTHEDOCS_GIT_BRANCH / GITHUB_REF_NAME →
#           "master".
import os


def _detect_current_branch() -> str:
    """Return a branch/commit identifier for GitHub links with minimal logic.

    Order: _GITHUB_BRANCH override → READTHEDOCS_GIT_IDENTIFIER (commit) →
    READTHEDOCS_GIT_BRANCH / GITHUB_REF_NAME → "master".
    """
    return (
        os.getenv("_GITHUB_BRANCH")
        or os.getenv("READTHEDOCS_GIT_IDENTIFIER")
        or os.getenv("READTHEDOCS_GIT_BRANCH")
        or os.getenv("GITHUB_REF_NAME")
        or "master"
    )


_GITHUB_BRANCH = _detect_current_branch()

# Define an extlink so we can write :ghfile:`examples/foo.py` in .rst/.md
extlinks = {
    "ghfile": (
        f"https://github.com/waltsims/k-wave-python/blob/{_GITHUB_BRANCH}/%s",
        "",  # leave default prefix empty so captions can be provided inline
    ),
    "ghdir": (
        f"https://github.com/waltsims/k-wave-python/tree/{_GITHUB_BRANCH}/%s",
        "",
    ),
}

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["README.md", "_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/waltsims/k-wave-python",
    "source_branch": _GITHUB_BRANCH,
    "source_directory": "docs/",
}

html_static_path = ["_static"]

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# Remove module name from object names
add_module_names = False
