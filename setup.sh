#!/usr/bin/env bash
# [SETUP] One command on the machine that RUNS the agent (Linux/macOS):   bash setup.sh
#   - creates .venv (falls back to `pip --user` if venv is unavailable)
#   - installs packages, preferring resources/wheels if fetch_resources.sh pushed them
#   - installs Ollama WITHOUT sudo into ~/.local/ollama if it is missing (Linux x86_64)
#   - starts `ollama serve`, pulls models, dumps API docs, checks everything, builds chroma_db/
# Override:  CODER=qwen2.5-coder:1.5b VLM=moondream bash setup.sh
#            OFFLINE=1 bash setup.sh      (pip uses resources/wheels only)
#            NO_VENV=1 bash setup.sh      (use the system Python)
set -euo pipefail
cd "$(dirname "$0")"

CODER="${CODER:-qwen2.5-coder:3b}"
EMBED="${EMBED:-nomic-embed-text}"
VLM="${VLM:-qwen2.5vl:3b}"
PY="${PYTHON:-python3}"
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

echo "== 1/7 Python environment"
PIP_USER=()
if [[ -z "${NO_VENV:-}" ]] && "$PY" -m venv .venv >/dev/null 2>&1; then
  PY="$(pwd)/.venv/bin/python"
  echo "using .venv  (activate later with: source .venv/bin/activate)"
else
  warn "no venv (install python3-venv to get one); using $PY with --user"
  PIP_USER=(--user)
  "$PY" -m pip install --help | grep -q break-system-packages && PIP_USER+=(--break-system-packages)
fi
"$PY" -m pip install -q --upgrade pip ${PIP_USER[@]+"${PIP_USER[@]}"} || true

echo "== 2/7 Python packages"
PIP_SRC=()
if ls resources/wheels/*.whl >/dev/null 2>&1; then
  PIP_SRC=(--find-links resources/wheels)
  [[ -n "${OFFLINE:-}" ]] && PIP_SRC+=(--no-index)
  echo "using pushed wheels in resources/wheels"
fi
"$PY" -m pip install ${PIP_USER[@]+"${PIP_USER[@]}"} ${PIP_SRC[@]+"${PIP_SRC[@]}"} -r requirements.txt

echo "== 3/7 Ollama"
if ! command -v ollama >/dev/null 2>&1 && [[ -x "$HOME/.local/ollama/bin/ollama" ]]; then
  export PATH="$HOME/.local/ollama/bin:$PATH"
fi
if ! command -v ollama >/dev/null 2>&1; then
  if [[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]]; then
    echo "Ollama not found: installing without sudo into ~/.local/ollama (about 1.5 GB, includes GPU libraries)"
    mkdir -p "$HOME/.local/ollama"
    url="https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst"
    tmp="$HOME/.local/ollama/ollama.tar.zst"
    curl -fL --retry 3 -o "$tmp" "$url"
    if tar --zstd -xf "$tmp" -C "$HOME/.local/ollama" 2>/dev/null; then :; else
      "$PY" -m pip install -q ${PIP_USER[@]+"${PIP_USER[@]}"} zstandard          # no zstd binary: unpack in Python
      "$PY" - "$tmp" "$HOME/.local/ollama" <<'EOF'
import sys, tarfile, zstandard
with open(sys.argv[1], "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r:
    with tarfile.open(fileobj=r, mode="r|") as t:
        t.extractall(sys.argv[2])
EOF
    fi
    rm -f "$tmp"
    export PATH="$HOME/.local/ollama/bin:$PATH"
    grep -q '.local/ollama/bin' "$HOME/.bashrc" 2>/dev/null || \
      echo 'export PATH="$HOME/.local/ollama/bin:$PATH"' >> "$HOME/.bashrc"
  else
    echo "Install Ollama from https://ollama.com/download and re-run."; exit 1
  fi
fi
ollama --version || true
if ! ollama list >/dev/null 2>&1; then
  echo "starting ollama serve in the background (log: ~/ollama.log)"
  nohup ollama serve >"$HOME/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done
fi

echo "== 4/7 Models: $CODER $EMBED $VLM"
for m in "$CODER" "$EMBED" "$VLM"; do
  if ollama list | awk 'NR>1{print $1}' | grep -qxE "$m(:latest)?"; then echo "  have $m"
  else ollama pull "$m" || warn "pull $m failed"; fi
done

echo "== 5/7 API docs from installed libraries -> docs/api/"
"$PY" build_corpus.py docs/api

echo "== 6/7 Environment check"
"$PY" check_env.py --coder "$CODER" --embed "$EMBED" --vlm "$VLM" || warn "fix the FAIL lines above"

echo "== 7/7 Build the vector store (chroma_db/) from docs/ (examples, api, external)"
if ollama list | awk 'NR>1{print $1}' | grep -qxE "$EMBED(:latest)?"; then
  echo "embedding $(find docs -type f \( -name '*.md' -o -name '*.py' -o -name '*.txt' \) | wc -l) files" \
       "(a few minutes on CPU, much faster on GPU)"
  "$PY" rag_agent.py --docs docs --embed "$EMBED" --rebuild
else
  warn "embedding model $EMBED missing: skipped. After 'ollama pull $EMBED' run:"
  warn "  $PY rag_agent.py --docs docs --embed $EMBED --rebuild"
fi

echo
echo "Ready. Next:"
echo "  source .venv/bin/activate 2>/dev/null; export PATH=\"\$HOME/.local/ollama/bin:\$PATH\""
echo "  python rag_agent.py --docs docs --task-file tasks/01_edges.txt --model $CODER"
echo "  python smoke_test.py --model $CODER"
