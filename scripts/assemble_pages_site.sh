#!/usr/bin/env bash
# Build Sphinx docs and assemble the GitHub Pages site:
#   site/       -> foveated player (repo root on Pages)
#   site/docs/  -> Sphinx HTML output
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/clean_docs.sh
python scripts/generate_docs.py
python scripts/build_static_demo_run.py --output-dir web/foveated-player/runs/example

rm -rf site
mkdir -p site
cp -r web/foveated-player/. site/
cp -r docs/_build/html site/docs
touch site/.nojekyll

test -f site/index.html
test -f site/docs/index.html
echo "Assembled GitHub Pages site at ${ROOT}/site"
