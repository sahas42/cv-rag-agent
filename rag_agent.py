#!/usr/bin/env python3
# [RAG-AGENT] rag_agent.py - offline RAG + self-correcting code agent for Computer Vision tasks
#
# Usage:
#   python rag_agent.py --docs ./docs --task "Detect edges in input.jpg with Canny and save edges.png"
#   python rag_agent.py --docs ./docs --task-file task.txt --image input.jpg --max-iters 6
#   python rag_agent.py --docs ./docs --rebuild            # re-index docs only
#
# Needs (all local): `ollama serve` running, models pulled/created beforehand:
#   coder model  (default qwen2.5-coder:3b)
#   embed model  (default nomic-embed-text)
#   optional VLM (e.g. qwen2.5vl:3b / moondream / llava-phi3) for --vlm

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- config
DEFAULT_CODER = "qwen2.5-coder:3b"
DEFAULT_EMBED = "nomic-embed-text"
DEFAULT_VLM = "qwen2.5vl:3b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):                    # OLLAMA_HOST may be "host:port"
    OLLAMA_URL = "http://" + OLLAMA_URL
DOC_EXTS = {".txt", ".md", ".rst", ".py", ".html", ".htm"}

# make sure nothing tries to reach the internet
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # chromadb telemetry off
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def log(msg):
    print(f"[agent {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. load + chunk docs
def load_docs(docs_dir):
    """Read every text-like file under docs_dir into LangChain Documents (no extra deps)."""
    docs = []
    for p in sorted(Path(docs_dir).rglob("*")):
        if not p.is_file() or p.suffix.lower() not in DOC_EXTS:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if p.suffix.lower() in {".html", ".htm"}:
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[ \t]+", " ", text)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": str(p), "ext": p.suffix.lower()}))
    return docs


def split_docs(docs, chunk_size=1200, chunk_overlap=150):
    py_split = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    md_split = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    txt_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for d in docs:
        ext = d.metadata.get("ext")
        splitter = py_split if ext == ".py" else md_split if ext in {".md", ".rst"} else txt_split
        chunks.extend(splitter.split_documents([d]))
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks


# ---------------------------------------------------------------- 2. vector store
def get_vectorstore(docs_dir, db_dir, embeddings, rebuild=False, collection="cv_docs"):
    from langchain_chroma import Chroma
    if rebuild and Path(db_dir).exists():
        import shutil
        shutil.rmtree(db_dir)
    vs = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=db_dir,
                collection_metadata={"hnsw:space": "cosine"})
    if vs._collection.count() == 0:                      # empty -> index the docs
        chunks = split_docs(load_docs(docs_dir))
        if not chunks:
            sys.exit(f"No documents found under {docs_dir}")
        log(f"indexing {len(chunks)} chunks into {db_dir} ...")
        batch = 64                                       # small batches = steady memory
        for i in range(0, len(chunks), batch):
            vs.add_documents(chunks[i:i + batch], ids=[f"c{j}" for j in range(i, i + len(chunks[i:i + batch]))])
    log(f"vector store ready: {vs._collection.count()} chunks")
    return vs


def retrieve(vs, query, k=6):
    docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=max(20, 3 * k))
    return docs


def format_context(docs, max_chars=6000):
    out, total = [], 0
    for d in docs:
        block = f"### {Path(d.metadata.get('source', '?')).name}\n{d.page_content.strip()}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n".join(out)


# ---------------------------------------------------------------- 3. prompts
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Python computer-vision engineer working OFFLINE.
    Write ONE complete, executable Python 3 script that solves the task.
    Rules:
    - Use only these libraries if needed: cv2 (OpenCV), numpy, PIL, skimage, scipy, matplotlib, standard library.
    - Prefer the API usage shown in the DOCUMENTATION section; do not invent functions.
    - No GUI: never call cv2.imshow, cv2.waitKey, plt.show or input(). Save results to files instead.
    - If an input image is missing, create a synthetic test image with numpy so the script still runs.
    - Check that cv2.imread did not return None.
    - Print a short summary of the results (shapes, counts, output file names) at the end.
    - Output ONLY the code inside a single ```python ... ``` block. No explanation.""")

FIRST_TEMPLATE = """DOCUMENTATION:
{context}

TASK:
{task}
{image_note}
Write the full script now."""

FENCE = "`" * 3
FIX_TEMPLATE = ("DOCUMENTATION:\n{context}\n\nTASK:\n{task}\n{image_note}\n"
                "Your previous script:\n" + FENCE + "python\n{code}\n" + FENCE + "\n\n"
                "It FAILED with this error:\n" + FENCE + "\n{error}\n" + FENCE + "\n{hint}\n"
                "Find the cause of the error, fix it, and return the COMPLETE corrected script "
                "(not a diff) in a single " + FENCE + "python block.")

HINTS = [
    (r"NoneType.*(shape|has no attribute)|_src.empty",
     "cv2.imread returned None: wrong path or missing file. Check `if img is None` and fall back "
     "to a synthetic numpy image."),
    (r"depth.*CV_8U|Unsupported depth|unsupported format",
     "Convert the array to uint8 first: img = np.clip(img, 0, 255).astype(np.uint8)."),
    (r"scn is 1|Invalid number of channels|cn == 3",
     "Channel mismatch: use cv2.cvtColor only on 3-channel BGR; grayscale images have 2 dims."),
    (r"(not enough|too many) values to unpack \(expected [23], got [23]\)",
     "OpenCV 4/5 findContours returns 2 values: contours, hierarchy = cv2.findContours(...)."),
    (r"Overload resolution failed|missing required argument",
     "Wrong arguments to an OpenCV function. Re-read its signature in the DOCUMENTATION."),
    (r"has no attribute",
     "That function/attribute does not exist in this version. Use one shown in the DOCUMENTATION."),
    (r"No module named|not installed",
     "That library is not installed. Use only cv2, numpy, PIL, skimage, scipy, matplotlib."),
    (r"Timeout",
     "Remove cv2.imshow/cv2.waitKey/plt.show/input() and any infinite loop."),
    (r"broadcast|not aligned|mismatch in its core",
     "Array shape mismatch: print shapes and resize/reshape so they match."),
    (r"must be (odd|positive)|ksize",
     "Kernel sizes must be positive odd integers, e.g. (5, 5)."),
    (r"integer argument expected|cannot be interpreted as an integer|must be integer",
     "Point coordinates and sizes must be ints: use int(round(x))."),
]


def hint_for(error):
    for pattern, hint in HINTS:
        if re.search(pattern, error, re.I):
            return "HINT: " + hint
    return ""


# ---------------------------------------------------------------- 4. code extraction
CODE_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_code(reply):
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)     # Qwen3 thinking
    blocks = CODE_RE.findall(reply)
    if blocks:
        return max(blocks, key=len).strip()
    if "```" in reply:                                   # unterminated fence
        return reply.split("```", 1)[1].split("\n", 1)[-1].strip()
    return reply.strip()                                 # model returned bare code


# ---------------------------------------------------------------- 5. execution
GUI_CALLS = [r"^\s*cv2\.imshow\(.*$", r"^\s*cv2\.waitKey\(.*$",
             r"^\s*cv2\.destroyAllWindows\(.*$", r"^\s*plt\.show\(.*$"]


def strip_gui(code):
    """Replace blocking GUI calls with `pass` instead of wasting an iteration on a timeout."""
    for p in GUI_CALLS:
        code = re.sub(p, lambda m: re.match(r"\s*", m.group(0)).group(0) + "pass  # removed GUI call",
                      code, flags=re.M)
    return code


def static_check(code):
    """Syntax errors, uninstalled imports and input() - caught without running anything."""
    import ast
    import importlib.util
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})\n{(e.text or '').rstrip()}"
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        for m in mods:
            if m and importlib.util.find_spec(m) is None:
                problems.append(f"ModuleNotFoundError: No module named '{m}' (not installed)")
    if re.search(r"\binput\(", code):
        problems.append("Do not call input(); hard-code values instead.")
    return "\n".join(problems) or None


def run_code(code, workdir, timeout=60, attempt=0):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / f"attempt_{attempt}.py"
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen", PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                              text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return False, out, (f"TimeoutError: script did not finish in {timeout}s. "
                            "Remove infinite loops, GUI calls (imshow/waitKey) and input().")
    ok = proc.returncode == 0 and "Traceback (most recent call last)" not in proc.stderr
    return ok, proc.stdout, proc.stderr


def trim_error(stderr, max_lines=25):
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    if not lines:
        return "Process exited with a non-zero status but printed no error."
    # keep the traceback tail (the useful part) and drop noisy warnings above it
    if "Traceback (most recent call last):" in lines:
        lines = lines[len(lines) - 1 - lines[::-1].index("Traceback (most recent call last):"):]
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------- 6. optional VLM check
def vlm_check(image_path, task, vlm_model):
    import ollama
    prompt = (f"The following image is the output of a program for this task:\n{task}\n"
              "Does the image look like a correct result? Answer YES or NO on the first line, "
              "then one sentence of reason.")
    r = ollama.Client(host=OLLAMA_URL).chat(model=vlm_model, messages=[{"role": "user", "content": prompt,
                                                 "images": [str(image_path)]}],
                    options={"temperature": 0})
    answer = r["message"]["content"].strip()
    return answer.upper().startswith("YES"), answer


# ---------------------------------------------------------------- 7. agent loop
def solve(task, llm, vs, workdir="agent_runs", max_iters=5, timeout=60, k=6,
          image=None, vlm_model=None):
    image_note = f"\nInput image path: {image}\n" if image else ""
    context = format_context(retrieve(vs, task, k=k))
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=FIRST_TEMPLATE.format(context=context, task=task,
                                                           image_note=image_note))]
    code, error = "", ""
    for it in range(1, max_iters + 1):
        log(f"iteration {it}/{max_iters}: asking the LLM ...")
        t_start = time.time()
        reply = llm.invoke(messages).content
        code = strip_gui(extract_code(reply))

        error = static_check(code)
        if error is None:
            ok, out, err = run_code(code, workdir, timeout=timeout, attempt=it)
            if ok:
                log("execution succeeded")
                if out.strip():
                    print(out.strip()[-2000:])
                if vlm_model:
                    outs = sorted((p for p in Path(workdir).glob("*.png") if p.stat().st_mtime >= t_start),
                                  key=os.path.getmtime)
                    if outs:
                        good, why = vlm_check(outs[-1], task, vlm_model)
                        log(f"VLM check on {outs[-1].name}: {why}")
                        if not good:
                            error = f"The script ran, but the output image looks wrong: {why}"
                if not error:
                    final = Path(workdir) / "solution.py"
                    final.write_text(code, encoding="utf-8")
                    log(f"saved {final}")
                    return code
            else:
                error = trim_error(err)
        log(f"failed:\n{textwrap.indent(error, '    ')}")

        # retrieve again using the error so the fix prompt carries the relevant API docs
        err_lines = [l for l in error.splitlines() if re.search(r"Error|error:|Exception", l)]
        err_query = task + "\n" + (err_lines[-1] if err_lines else error.splitlines()[-1])
        context = format_context(retrieve(vs, err_query, k=k))
        messages = [SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=FIX_TEMPLATE.format(context=context, task=task, code=code,
                                                             error=error, image_note=image_note,
                                                             hint=hint_for(error)))]
    log(f"gave up after {max_iters} iterations; last code saved as {workdir}/last_failed.py")
    (Path(workdir) / "last_failed.py").write_text(code, encoding="utf-8")
    return None


# ---------------------------------------------------------------- 8. CLI
def main():
    ap = argparse.ArgumentParser(description="Offline RAG code agent for CV tasks")
    ap.add_argument("--docs", default="docs", help="folder with API documentation files")
    ap.add_argument("--db", default="chroma_db", help="Chroma persist directory")
    ap.add_argument("--task", help="natural-language CV task")
    ap.add_argument("--task-file", help="read the task from a text file")
    ap.add_argument("--image", help="input image path passed to the generated code")
    ap.add_argument("--model", default=DEFAULT_CODER)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--vlm", default=None, help=f"VLM to judge output images, e.g. {DEFAULT_VLM}")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workdir", default="agent_runs")
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the vector store")
    args = ap.parse_args()

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=args.embed, base_url=OLLAMA_URL)
    vs = get_vectorstore(args.docs, args.db, embeddings, rebuild=args.rebuild)

    task = Path(args.task_file).read_text() if args.task_file else args.task
    if not task:
        return
    if args.image:
        args.image = str(Path(args.image).resolve())    # absolute: code runs inside workdir
    llm = ChatOllama(model=args.model, base_url=OLLAMA_URL, temperature=0.2,
                     num_ctx=8192, num_predict=2048)
    code = solve(task, llm, vs, workdir=args.workdir, max_iters=args.max_iters,
                 timeout=args.timeout, k=args.k, image=args.image, vlm_model=args.vlm)
    sys.exit(0 if code else 1)


if __name__ == "__main__":
    main()
