#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${LC_ALL:-}" == "C.UTF-8" || -z "${LC_ALL:-}" ]]; then
  export LC_ALL="en_US.UTF-8"
fi

if [[ "${LANG:-}" == "C.UTF-8" || -z "${LANG:-}" ]]; then
  export LANG="en_US.UTF-8"
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Error: latexmk no esta instalado o no esta en el PATH." >&2
  exit 1
fi

if [[ "${1:-}" == "--clean" || "${1:-}" == "-c" ]]; then
  latexmk -C main.tex
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "PDF generado en: $SCRIPT_DIR/main.pdf"
