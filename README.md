# cv-rag-agent

An offline RAG code agent for Computer Vision tasks. Given a natural-language CV task, it
retrieves relevant API documentation from a local ChromaDB index, asks a small local LLM
(via Ollama) for a complete Python script, runs it in a subprocess, and feeds any error back
to the LLM until the script runs cleanly. An optional vision model judges the output image.

```
docs/ ──load+chunk──> embed (nomic-embed-text) ──> chroma_db/
task ──> retrieve (MMR) ──> prompt ──> coder LLM ──> extract code ──> static check + run
                ^                                                         │
                └──────────── no: traceback + targeted hint ──────────────┘
                                                      yes ──> agent_runs/solution.py
```

No internet is needed at run time: models, embeddings and docs are all local.

## Quick start (exam machine)

```bash
git clone <your-repo-url> cv-rag-agent && cd cv-rag-agent
bash setup.sh                 # Windows: .\setup.ps1
python rag_agent.py --docs docs --task "Detect edges in input.jpg with Canny and save edges.png" --image input.jpg
```

`setup.sh` installs packages, starts Ollama, pulls the three models if missing, dumps API docs
of the installed libraries into `docs/api/`, checks everything, and builds `chroma_db/`.
Different models: `CODER=qwen2.5-coder:1.5b VLM=moondream bash setup.sh`.

## Exam-day checklist

1. `python check_env.py` (add `--vlm qwen2.5vl:3b` if you use one). Every line must say PASS.
2. Note the versions it prints; if the provided models differ, pass `--model` / `--embed`.
3. If you were given a docs folder, copy it into `docs/` (or point `--docs` at it) and run
   `python rag_agent.py --docs docs --rebuild`.
4. `python smoke_test.py --model <coder>` to see speed and success rate on 5 sample tasks.
5. Solve the real task; adjust `--max-iters`, `--timeout`, `--k` as needed.

## Commands

| Command | What it does |
| --- | --- |
| `python check_env.py [--coder M] [--embed E] [--vlm V]` | Packages, server, models, one chat/embed/vision call |
| `python build_corpus.py [out_dir] [modules...]` | Docstrings of installed libs -> Markdown (default `docs/api/`) |
| `python rag_agent.py --docs docs --rebuild` | (Re)build the vector store only |
| `python rag_agent.py --docs docs --task "..." [--image x.jpg]` | Solve a task |
| `python rag_agent.py --docs docs --task-file tasks/02_count_objects.txt` | Task from a file |
| `python rag_agent.py ... --vlm qwen2.5vl:3b` | Also require a VLM to say the output image looks right |
| `python smoke_test.py [--model M] [--fake]` | Run every task in `tasks/`, print a results table |
| `python tests/fake_ollama_server.py --port 11435` | Ollama-compatible fake server for rehearsing |
| `python -m pytest -q` | 20 offline tests (no models needed) |

`rag_agent.py` options: `--model` (default `qwen2.5-coder:3b`), `--embed` (`nomic-embed-text`),
`--vlm`, `--max-iters` (5), `--timeout` seconds per run (60), `--k` chunks retrieved (6),
`--db` (`chroma_db`), `--workdir` (`agent_runs`). Exit code 0 = solved
(`agent_runs/solution.py`), 1 = gave up (`agent_runs/last_failed.py`). Ollama address comes
from `OLLAMA_HOST` (default `http://127.0.0.1:11434`).

## Files

| Path | Purpose |
| --- | --- |
| `rag_agent.py` | The agent: load/chunk docs, Chroma index, MMR retrieval, prompts, code extraction, static check, subprocess run, error hints, fix loop, optional VLM check, CLI |
| `build_corpus.py` | Turns installed `cv2`, `numpy`, `skimage`, `PIL` docstrings into `docs/api/*.md` |
| `check_env.py` | Pre-flight check of packages, Ollama and models |
| `smoke_test.py` | Runs the agent on all sample tasks with a synthetic input image |
| `docs/examples/` | 27 tested OpenCV / NumPy / scikit-image examples, retrieved by the agent |
| `tasks/` | Sample CV tasks |
| `modelfiles/` | Templates to import a local `.gguf` (Qwen, Mistral, LLaVA) with `ollama create` |
| `tests/test_agent.py` | Offline tests: extraction, checks, hints, runner, fix loop, CLI over HTTP, every example |
| `tests/fake_ollama_server.py` | Fake Ollama API: buggy code first, fixed code after "It FAILED" |
| `requirements.txt` / `requirements-lock.txt` | Minimum versions / exact versions tested |

## How it was tested

* All 20 tests pass on two stacks:
  LangChain 1.x (core 1.7, langchain-ollama 1.1, langchain-chroma 1.1), chromadb 1.5, ollama 0.6, OpenCV 5.0, NumPy 2.5;
  and LangChain 0.3 (core 0.3, langchain-ollama 0.3, langchain-chroma 0.2), chromadb 0.6, ollama 0.4, OpenCV 4.10, NumPy 2.2.
* The end-to-end tests run the real CLI over HTTP against the fake Ollama server, covering
  `ChatOllama` streaming, `OllamaEmbeddings`, Chroma persistence, a failing first attempt,
  a successful fix, and the VLM check.
* Not tested with real model weights (the build sandbox could not download them):
  run `python smoke_test.py` once on your machine to measure real speed and success rate.

## Using a model without Ollama

* Raw GGUF in Ollama: put it in `models/`, then `ollama create my-coder -f modelfiles/qwen-chatml.Modelfile`
  and pass `--model my-coder`.
* llama-cpp-python: replace the `ChatOllama(...)` line in `main()` with any object whose
  `.invoke(messages).content` returns text (see the manual's `[LLAMACPP-SWAP]`).
* A vLLM / OpenAI-compatible server: `from langchain_openai import ChatOpenAI` and
  `ChatOpenAI(base_url="http://localhost:8000/v1", api_key="none", model="<served name>")`.

## Common fixes

| Symptom | Fix |
| --- | --- |
| `Connection refused` | `ollama serve` |
| `model "x" not found` | exact tag from `ollama list` |
| `Collection expecting embedding with dimension ...` | changed embed model: add `--rebuild` |
| Model output cut off | raise `num_predict` in `main()` |
| Same error every iteration | raise `--max-iters`; add a hint to `HINTS` in `rag_agent.py` |
| Slow on CPU | `--model qwen2.5-coder:1.5b`, `--k 4`, `OLLAMA_KEEP_ALIVE=30m` |
