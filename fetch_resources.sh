#!/usr/bin/env bash
# [FETCH-RESOURCES] Fetch docs, test images, wheels and tools on the DEV DESKTOP, then push
# them to the REMOTE RUN MACHINE (or a USB drive) and bootstrap it there.
#
#   bash fetch_resources.sh                         # desktop: docs + images + wheels + vscode
#   bash fetch_resources.sh docs                    # just the docs
#   bash fetch_resources.sh push --to user@host:~/cv-rag-agent [--port 22] [--key ~/.ssh/id_ed25519]
#   bash fetch_resources.sh remote --to user@host:~/cv-rag-agent        # run setup on the remote
#   bash fetch_resources.sh all  --to user@host:~/cv-rag-agent          # everything, end to end
#   bash fetch_resources.sh push --to /media/usb/cv-rag-agent           # copy to a USB drive
#
# Steps:  docs images wheels vscode models push remote   (default: docs images wheels vscode)
# Env:    REMOTE_PY=3.11  REMOTE_PLAT=linux|win|mac  (wheels are built for the REMOTE machine)
#         CODER=qwen2.5-coder:3b EMBED=nomic-embed-text VLM=qwen2.5vl:3b  FALLBACK_MODELS="..."
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# ------------------------------------------------------------------ config
CODER="${CODER:-qwen2.5-coder:3b}"
EMBED="${EMBED:-nomic-embed-text}"
VLM="${VLM:-qwen2.5vl:3b}"
FALLBACK_MODELS="${FALLBACK_MODELS:-qwen2.5-coder:1.5b moondream all-minilm}"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
REMOTE_PY="${REMOTE_PY:-$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')}"
REMOTE_PLAT="${REMOTE_PLAT:-linux}"
RES="$ROOT/resources"                 # wheels, images, repo cache (not committed)
EXT="$ROOT/docs/external"             # third-party CV docs, indexed by rag_agent.py (not committed)
REF="$ROOT/reference"                  # docs for YOU (Ollama/LangChain/Chroma), not indexed
TO=""; PORT=22; KEY=""
STEPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to) TO="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    all) STEPS+=(docs images wheels vscode push remote); shift ;;
    docs|images|wheels|vscode|models|push|remote) STEPS+=("$1"); shift ;;
    *) echo "unknown argument: $1 (see --help)"; exit 2 ;;
  esac
done
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=(docs images wheels vscode)

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }
SSH_OPTS=(-p "$PORT" -o ServerAliveInterval=30); [[ -n "$KEY" ]] && SSH_OPTS+=(-i "$KEY")
is_remote() { [[ "$TO" == *:* && "$TO" != /* ]]; }       # user@host:dir  vs  /local/path
remote_host() { echo "${TO%%:*}"; }
remote_dir()  { local d="${TO#*:}"; echo "${d:-cv-rag-agent}"; }

# ------------------------------------------------------------------ docs
# sparse_clone <name> <github owner/repo> <path>...   -> resources/repos/<name> (shallow, only those paths)
sparse_clone() {
  local name="$1" repo="$2"; shift 2
  local dir="$RES/repos/$name"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" sparse-checkout set "$@" && git -C "$dir" pull -q --depth 1 || warn "update of $name failed, using cache"
  else
    git clone -q --depth 1 --filter=blob:none --sparse "https://github.com/$repo.git" "$dir"
    git -C "$dir" sparse-checkout set "$@"
  fi
  echo "$name  https://github.com/$repo  $(git -C "$dir" rev-parse --short HEAD)" >> "$REF/SOURCES.txt"
}

# copy_text <src_dir> <dest_dir> [find -name patterns...]: copy text docs, renaming .mdx/.markdown/.rst to .md
copy_text() {
  local src="$1" dst="$2"; shift 2
  [[ -d "$src" ]] || { warn "missing $src"; return 0; }
  local pats=("$@"); [[ ${#pats[@]} -eq 0 ]] && pats=('*.md' '*.mdx' '*.markdown' '*.rst' '*.py' '*.txt')
  local expr=(); for p in "${pats[@]}"; do expr+=(-o -name "$p"); done
  while IFS= read -r -d '' f; do
    local rel="${f#"$src"/}" out
    out="$dst/${rel%.*}"
    case "$f" in *.py|*.txt) out="$out.${f##*.}" ;; *) out="$out.md" ;; esac
    mkdir -p "$(dirname "$out")"
    if [[ "$f" == *.mdx ]]; then           # drop MDX import/export lines; keep prose and code
      grep -vE '^(import|export) ' "$f" > "$out" || true
    else
      cp "$f" "$out"
    fi
  done < <(find "$src" -type f \( -false "${expr[@]}" \) -print0)
}

step_docs() {
  log "docs: CV docs -> $EXT (indexed), dev docs -> $REF (for you)"
  command -v git >/dev/null || { warn "git not found"; return 1; }
  mkdir -p "$RES/repos" "$EXT" "$REF"; : > "$REF/SOURCES.txt"
  local R="$RES/repos"

  sparse_clone opencv opencv/opencv doc/py_tutorials samples/python samples/data
  copy_text "$R/opencv/doc/py_tutorials" "$EXT/opencv/py_tutorials" '*.markdown' '*.md'
  copy_text "$R/opencv/samples/python"   "$EXT/opencv/samples" '*.py'

  sparse_clone scikit-image scikit-image/scikit-image doc/examples
  copy_text "$R/scikit-image/doc/examples" "$EXT/skimage/examples" '*.py' '*.txt' '*.rst'

  sparse_clone ollama ollama/ollama docs
  copy_text "$R/ollama/docs" "$REF/ollama"

  sparse_clone langchain langchain-ai/docs src/oss/langchain src/oss/integrations/splitters \
      src/oss/python/integrations/chat src/oss/python/integrations/embeddings \
      src/oss/python/integrations/llms src/oss/python/integrations/vectorstores \
      src/oss/python/integrations/providers
  copy_text "$R/langchain/src/oss/langchain"              "$REF/langchain/core"
  copy_text "$R/langchain/src/oss/integrations/splitters" "$REF/langchain/splitters"
  for kind in chat embeddings llms vectorstores providers; do    # only the integrations we use
    copy_text "$R/langchain/src/oss/python/integrations/$kind" "$REF/langchain/$kind" \
              'ollama.mdx' 'chroma.mdx' 'llamacpp.mdx' 'huggingface*.mdx'
  done

  sparse_clone chroma chroma-core/chroma docs/mintlify/docs
  copy_text "$R/chroma/docs/mintlify/docs" "$REF/chroma"

  echo "$(find "$EXT" -type f | wc -l) files, $(du -sh "$EXT" | cut -f1) in docs/external  (OpenCV + scikit-image, indexed)"
  echo "$(find "$REF" -type f | wc -l) files, $(du -sh "$REF" | cut -f1) in reference/  (Ollama, LangChain, Chroma; sources in reference/SOURCES.txt)"
  echo "Search the reference docs:  grep -rn \"with_structured_output\" reference/"
  echo "Rebuild the agent index:    $PY rag_agent.py --docs docs --rebuild"
}

# ------------------------------------------------------------------ test images
step_images() {
  log "images -> $RES/images"
  mkdir -p "$RES/images"
  local cv="$RES/repos/opencv/samples/data"
  if [[ -d "$cv" ]]; then
    find "$cv" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.avi' -o -iname '*.mp4' \) \
         -size -6M -exec cp {} "$RES/images/" \;
  else
    warn "run the docs step first to get OpenCV sample images"
  fi
  "$PY" - "$RES/images" <<'EOF' || warn "scikit-image not installed: skipped its sample images"
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from skimage import data, io, util
out = Path(sys.argv[1])
for name in ["astronaut", "camera", "coins", "coffee", "chelsea", "text", "page", "moon",
             "horse", "rocket", "checkerboard", "clock", "hubble_deep_field", "immunohistochemistry"]:
    try:
        img = getattr(data, name)()
        if img.dtype == bool:
            img = util.img_as_ubyte(img)
        io.imsave(str(out / f"skimage_{name}.png"), img, check_contrast=False)
    except Exception as e:                       # some need a download; skip quietly
        print(f"  skip {name}: {type(e).__name__}")
print(f"  saved skimage sample images to {out}")
EOF
  echo "$(ls "$RES/images" | wc -l) images in resources/images"
}

# ------------------------------------------------------------------ wheels for the REMOTE machine
step_wheels() {
  log "wheels for Python $REMOTE_PY on $REMOTE_PLAT -> $RES/wheels"
  mkdir -p "$RES/wheels"
  local plat=()
  case "$REMOTE_PLAT" in
    linux) plat=(--platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64
                 --platform manylinux_2_28_x86_64 --platform manylinux_2_34_x86_64 --platform linux_x86_64) ;;
    win)   plat=(--platform win_amd64) ;;
    mac)   plat=(--platform macosx_11_0_arm64 --platform macosx_12_0_arm64 --platform macosx_14_0_arm64) ;;
    *)     warn "REMOTE_PLAT must be linux, win or mac"; return 1 ;;
  esac
  "$PY" -m pip download -q -r requirements.txt -d "$RES/wheels" \
      --only-binary=:all: --python-version "$REMOTE_PY" "${plat[@]}" \
    || warn "some wheels failed; the remote can still pip install from the internet"
  echo "$(ls "$RES/wheels" | wc -l) files, $(du -sh "$RES/wheels" | cut -f1) in resources/wheels"
  echo "Remote install: pip install --no-index --find-links resources/wheels -r requirements.txt"
}

# ------------------------------------------------------------------ VS Code extensions (desktop)
step_vscode() {
  log "VS Code extensions"
  if ! command -v code >/dev/null 2>&1; then warn "'code' CLI not on PATH: skipped"; return 0; fi
  for ext in ms-python.python ms-python.vscode-pylance ms-python.debugpy ms-toolsai.jupyter \
             charliermarsh.ruff usernamehw.errorlens ms-vscode-remote.remote-ssh; do
    code --install-extension "$ext" --force >/dev/null && echo "  ok $ext" || warn "failed $ext"
  done
  echo "Remote - SSH lets you edit and run on the remote machine from this desktop."
}

# ------------------------------------------------------------------ models (where Ollama runs)
step_models() {
  log "Ollama models: $CODER $EMBED $VLM + fallbacks: $FALLBACK_MODELS"
  command -v ollama >/dev/null || { warn "ollama not installed here: skipped (the remote step pulls there)"; return 0; }
  for m in $CODER $EMBED $VLM $FALLBACK_MODELS; do ollama pull "$m" || warn "pull $m failed"; done
}

# ------------------------------------------------------------------ push to remote or USB
step_push() {
  [[ -n "$TO" ]] || { warn "push needs --to user@host:dir  or  --to /local/path"; return 1; }
  log "push repo + docs/external + resources -> $TO"
  local excl=("--exclude=.git" "--exclude=chroma_db*" "--exclude=agent_runs" "--exclude=smoke_runs"
              "--exclude=__pycache__" "--exclude=.pytest_cache" "--exclude=.venv" "--exclude=resources/repos")
  if is_remote; then
    ssh "${SSH_OPTS[@]}" "$(remote_host)" "mkdir -p $(remote_dir)"
    if command -v rsync >/dev/null && ssh "${SSH_OPTS[@]}" "$(remote_host)" "command -v rsync" >/dev/null 2>&1; then
      rsync -az --info=progress2 "${excl[@]}" -e "ssh ${SSH_OPTS[*]}" "$ROOT/" "$TO/"
    else                                   # no rsync (e.g. Git Bash on Windows): tar over ssh
      tar -C "$ROOT" "${excl[@]}" -czf - . \
        | ssh "${SSH_OPTS[@]}" "$(remote_host)" "tar -C $(remote_dir) -xzf -"
    fi
  else
    mkdir -p "$TO"
    if command -v rsync >/dev/null; then
      rsync -a --info=progress2 "${excl[@]}" "$ROOT/" "$TO/"
    else
      tar -C "$ROOT" "${excl[@]}" -cf - . | tar -C "$TO" -xf -
    fi
  fi
  echo "pushed to $TO"
}

# ------------------------------------------------------------------ bootstrap the remote
step_remote() {
  [[ -n "$TO" ]] || { warn "remote needs --to user@host:dir"; return 1; }
  if ! is_remote; then warn "--to is a local path: run 'bash setup.sh' there yourself"; return 0; fi
  log "remote setup on $(remote_host):$(remote_dir)"
  ssh -t "${SSH_OPTS[@]}" "$(remote_host)" \
    "cd $(remote_dir) && CODER='$CODER' EMBED='$EMBED' VLM='$VLM' bash setup.sh"
  echo "Then on the remote:  cd $(remote_dir) && python3 smoke_test.py --model $CODER"
}

for s in "${STEPS[@]}"; do "step_$s"; done
log "done: ${STEPS[*]}"
