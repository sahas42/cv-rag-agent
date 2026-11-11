#!/usr/bin/env bash
# [SETUP] One command on the exam machine (Linux/macOS):   bash setup.sh
# Override models:  CODER=qwen2.5-coder:1.5b VLM=moondream bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

CODER="${CODER:-qwen2.5-coder:3b}"
EMBED="${EMBED:-nomic-embed-text}"
VLM="${VLM:-qwen2.5vl:3b}"
PY="${PYTHON:-python3}"

echo "== 1/6 Python packages"
"$PY" -m pip install -r requirements.txt

echo "== 2/6 Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install it:  curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "starting ollama serve in the background (log: /tmp/ollama.log)"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 5
fi

echo "== 3/6 Models: $CODER $EMBED $VLM"
for m in "$CODER" "$EMBED" "$VLM"; do
  ollama list | awk '{print $1}' | grep -qx "$m" || ollama list | awk '{print $1}' | grep -qx "$m:latest" \
    || ollama pull "$m"
done

echo "== 4/6 API docs from installed libraries -> docs/api/"
"$PY" build_corpus.py docs/api

echo "== 5/6 Environment check"
"$PY" check_env.py --coder "$CODER" --embed "$EMBED" --vlm "$VLM"

echo "== 6/6 Build the vector store (chroma_db/)"
"$PY" rag_agent.py --docs docs --embed "$EMBED" --rebuild

echo
echo "Ready. Try:"
echo "  $PY rag_agent.py --docs docs --task-file tasks/01_edges.txt --model $CODER"
echo "  $PY smoke_test.py --model $CODER     # all sample tasks with real models"
