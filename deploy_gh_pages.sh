#!/usr/bin/env bash
# Deploy vllm-learning-html/ to a GitHub Pages site.
#
# Usage:
#   ./deploy_gh_pages.sh git@github.com:<user>/<repo>.git [branch]
#
# Behavior:
#   - Initializes a fresh git repo inside vllm-learning-html/ (gitignored otherwise)
#   - Commits everything, pushes to <branch> (default: gh-pages) on the given remote
#   - Force-push (this branch is a build artifact, not source of truth)
#
# Prereqs:
#   - Build HTML first: `python3 build_html.py`
#   - Optional: also build PDF/EPUB: `python3 build_pdf_epub.py`
#   - GitHub repo exists, Pages is configured to deploy from the chosen branch
#
# Tip: the target repo should NOT have Jekyll processing (the .nojekyll file is auto-emitted).

set -euo pipefail

REMOTE="${1:-}"
BRANCH="${2:-gh-pages}"
# Default to the sibling vllm-learning-html/ next to this script's directory,
# but allow VLLM_LEARNING_DST to override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_DIR="${VLLM_LEARNING_DST:-$SCRIPT_DIR/../vllm-learning-html}"

if [[ -z "$REMOTE" ]]; then
  cat <<EOF
Usage: $0 <remote-url> [branch]

Example:
  $0 git@github.com:yourname/vllm-learning-notes.git gh-pages

Or for a https remote with token:
  $0 https://<TOKEN>@github.com/yourname/vllm-learning-notes.git gh-pages

After pushing, enable GitHub Pages in the repo settings:
  Settings → Pages → Branch: $BRANCH / (root)

Site will be served at: https://<user>.github.io/<repo>/
EOF
  exit 1
fi

if [[ ! -d "$HTML_DIR" ]]; then
  echo "ERROR: $HTML_DIR not found. Run 'python3 build_html.py' first."
  exit 1
fi

if [[ ! -f "$HTML_DIR/index.html" ]]; then
  echo "ERROR: $HTML_DIR/index.html missing."
  exit 1
fi

cd "$HTML_DIR"

# Always start from a clean local repo state for this artifact directory
rm -rf .git

git init -q
git checkout -q -b "$BRANCH"

git add -A
git -c user.email="deploy@local" -c user.name="vllm-learning deploy" \
    commit -q -m "Publish vllm-learning ($(date +%Y-%m-%dT%H:%M:%S%z))"

echo "Pushing to $REMOTE branch $BRANCH ..."
git remote add origin "$REMOTE"
git push -f origin "$BRANCH"

echo ""
echo "Done. Enable Pages at: Settings → Pages → Branch: $BRANCH"
echo "Site URL pattern:      https://<user>.github.io/<repo>/"
