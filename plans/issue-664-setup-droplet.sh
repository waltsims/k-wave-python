#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 24.04 x86_64 droplet for issue #664 work.
#
# Usage on the droplet (as a sudo-capable user):
#   curl -fsSL https://raw.githubusercontent.com/waltsims/k-wave-python/fix-alpha-mode-near-unity/plans/issue-664-setup-droplet.sh | bash
# Or after cloning:
#   bash plans/issue-664-setup-droplet.sh
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/waltsims/k-wave-python.git}"
BRANCH="${BRANCH:-fix-alpha-mode-near-unity}"
WORKDIR="${WORKDIR:-$HOME/k-wave-python}"

echo "==> Installing system dependencies"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    libfftw3-dev libhdf5-dev zlib1g-dev libgomp1

echo "==> Installing uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> Installing GitHub CLI"
if ! command -v gh >/dev/null 2>&1; then
    type -p curl >/dev/null
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo dd of=/etc/apt/keyrings/githubcli-archive-keyring.gpg
    sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y gh
fi
gh --version | head -1

echo "==> Installing Claude Code (npm)"
if ! command -v claude >/dev/null 2>&1; then
    if ! command -v npm >/dev/null 2>&1; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    sudo npm install -g @anthropic-ai/claude-code
fi
claude --version || true

echo "==> Cloning repo"
if [ ! -d "$WORKDIR" ]; then
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "==> Syncing Python dependencies"
uv sync --extra all

echo
echo "=================================================================="
echo "Setup complete."
echo
echo "Next steps:"
echo "  1) gh auth login            # one-time, interactive"
echo "  2) cd $WORKDIR"
echo "  3) uv run pytest tests/test_issue_664_alpha_power_near_unity.py -v"
echo "     (smoke test should fail with NaN — that's expected on master)"
echo "  4) claude                   # then read plans/issue-664.md"
echo "=================================================================="
