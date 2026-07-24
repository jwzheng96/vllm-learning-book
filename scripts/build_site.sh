#!/usr/bin/env bash
# Assemble the MkDocs docs directory (_web/) from the book Markdown sources.
# Only Markdown + images are copied; build scripts, LaTeX, source-sync tools
# are left out so the generated site stays clean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/_web"

rm -rf "$DEST"
mkdir -p "$DEST"

# Site homepage.
cp "$ROOT/index.md" "$DEST/index.md"

# The 9 Part directories (01-overview ~ 09-advanced-features).
# Each contains tutorial .md files; README.md (Part nav pages) are excluded
# because mkdocs.yml nav already provides sidebar navigation.
for d in "$ROOT"/0*-*/; do
  [ -d "$d" ] || continue
  part=$(basename "$d")
  mkdir -p "$DEST/$part"
  for f in "$d"*.md; do
    [ -f "$f" ] || continue
    [ "$(basename "$f")" = "README.md" ] && continue
    # Strip leading "NN. " or "NNb. " numbering from H1 headings — the nav
    # sidebar already shows chapter ordering, so the manual prefix is noise.
    sed -E 's/^(#+ )[0-9]+[a-z]?\. /\1/' "$f" > "$DEST/$part/$(basename "$f")"
  done
done

# Site-level static assets (logo, favicon, OG cover).
if [ -d "$ROOT/assets" ]; then
  mkdir -p "$DEST/assets"
  cp -R "$ROOT/assets/." "$DEST/assets/"
fi

# Extra JS/CSS for mkdocs (mermaid init, theme tweaks).
if [ -d "$ROOT/extras" ]; then
  cp -R "$ROOT/extras" "$DEST/extras"
fi

# Keep only Markdown and web-safe assets; drop everything else.
find "$DEST" \( -type f -o -type l \) \
  ! -name '*.md' \
  ! -name '*.svg' \
  ! -name '*.png' \
  ! -name '*.jpg' \
  ! -name '*.jpeg' \
  ! -name '*.js' \
  ! -name '*.css' \
  ! -name '*.txt' \
  -delete

echo "Assembled _web/ ($(find "$DEST" -name '*.md' | wc -l | tr -d ' ') markdown files)"
