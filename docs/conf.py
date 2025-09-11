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

# Provide a smart GitHub branch fallback for linking to example files.
# Priority: explicit READTHEDOCS_GIT_BRANCH (when building on RTD) →
#          CI/CD branch via GITHUB_REF_NAME →
#          local git branch via `git rev-parse --abbrev-ref HEAD` (optional) →
#          "main".
import os
import subprocess


def _detect_current_branch() -> str:
    """
    Return the best-effort git branch name to use for documentation links.
    
    Checks, in order of precedence:
    1. READTHEDOCS_GIT_BRANCH environment variable
    2. GITHUB_REF_NAME environment variable
    3. Local git via `git rev-parse --abbrev-ref HEAD`
    
    If none of the above yield a branch name (or the git command fails), returns "main". The function swallows errors from the git probe.
    """
    for env_var in ("READTHEDOCS_GIT_BRANCH", "GITHUB_REF_NAME"):
        branch = os.getenv(env_var)
        if branch:
            return branch
    # Fallback: try local git (will fail on RTD but env vars cover that)
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        if branch:
            return branch
    except Exception:
        pass
    return "main"


_GITHUB_BRANCH = _detect_current_branch()

# Define an extlink so we can write :ghfile:`examples/foo.py` in .rst/.md
extlinks = {
    "ghfile": (
        f"https://github.com/waltsims/k-wave-python/blob/{_GITHUB_BRANCH}/%s",
        "",  # leave default prefix empty so captions can be provided inline
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
    "source_branch": "master",
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
