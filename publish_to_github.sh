#!/usr/bin/env bash
# Creates the dedicated repository in the user's account and pushes the app.
# Requires Git, GitHub CLI (gh), a logged-in gh account, and user Git identity.
set -euo pipefail
REPO_NAME="somnitech-living-review"
VISIBILITY="${1:-private}"
if [[ "$VISIBILITY" != "private" && "$VISIBILITY" != "public" ]]; then
  echo "Aufruf: ./publish_to_github.sh [private|public]" >&2; exit 2
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) fehlt. Installation: https://cli.github.com/" >&2; exit 1
fi
gh auth status >/dev/null
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  git init -b main
fi
if [[ -n "$(git status --porcelain)" ]]; then
  git add .
  git commit -m "Initial SomniTech Living Evidence Lab app"
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
else
  git push -u origin main
fi
echo "GitHub-Upload abgeschlossen."
