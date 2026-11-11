#!/usr/bin/env python3
# [CHECK-ENV] First thing to run on the exam machine.
# Checks Python packages, the Ollama server, the models, and does one tiny chat + embed call.
#
#   python check_env.py
#   python check_env.py --coder qwen2.5-coder:1.5b --embed all-minilm --vlm moondream
import argparse
import importlib
import os
import sys
import time

PACKAGES = ["langchain_core", "langchain_ollama", "langchain_chroma", "langchain_text_splitters",
            "chromadb", "ollama", "cv2", "numpy", "PIL", "skimage", "scipy", "matplotlib"]

ok_all = True


def report(ok, msg):
    global ok_all
    ok_all &= ok
    print(("  PASS " if ok else "  FAIL ") + msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder", default="qwen2.5-coder:3b")
    ap.add_argument("--embed", default="nomic-embed-text")
    ap.add_argument("--vlm", default=None, help="optional, e.g. qwen2.5vl:3b")
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    print("Packages:")
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            report(True, f"{name} {getattr(m, '__version__', '')}")
        except Exception as e:
            report(False, f"{name}: {e}  -> pip install -r requirements.txt")

    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    print(f"Ollama at {host}:")
    try:
        import ollama
        client = ollama.Client(host=host, timeout=600)
        names = [m.model for m in client.list().models]
        report(True, f"server reachable, {len(names)} models: {', '.join(names) or '(none)'}")
    except Exception as e:
        report(False, f"server not reachable ({e}) -> run `ollama serve`")
        sys.exit(1)

    def have(model):
        full = model if ":" in model else model + ":latest"
        return full in names or model in names

    wanted = [args.coder, args.embed] + ([args.vlm] if args.vlm else [])
    for model in wanted:
        report(have(model), f"model {model}" + ("" if have(model) else f" missing -> ollama pull {model}"))

    if have(args.coder):
        t = time.time()
        r = client.chat(model=args.coder, options={"temperature": 0, "num_predict": 64},
                        messages=[{"role": "user", "content": "Write a one-line Python print statement."}])
        report(bool(r.message.content.strip()), f"chat with {args.coder} in {time.time() - t:.1f}s: "
                                                 f"{r.message.content.strip()[:60]!r}")
    if have(args.embed):
        t = time.time()
        e = client.embed(model=args.embed, input=["cv2.Canny edge detection"])
        dim = len(e.embeddings[0]) if e.embeddings else 0
        report(dim > 0, f"embed with {args.embed}: dim {dim} in {time.time() - t:.1f}s")
    if args.vlm and have(args.vlm):
        import numpy as np
        import cv2
        img = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(img, (32, 32), 20, (0, 0, 255), -1)
        ok, buf = cv2.imencode(".png", img)
        t = time.time()
        r = client.chat(model=args.vlm, options={"temperature": 0, "num_predict": 40},
                        messages=[{"role": "user", "content": "What shape is in this image?",
                                   "images": [buf.tobytes()]}])
        report(bool(r.message.content.strip()), f"vision with {args.vlm} in {time.time() - t:.1f}s: "
                                                 f"{r.message.content.strip()[:60]!r}")

    print("\nALL GOOD" if ok_all else "\nSOME CHECKS FAILED (see FAIL lines above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
