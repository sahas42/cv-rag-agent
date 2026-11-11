#!/usr/bin/env python3
# [SMOKE-TEST] Run rag_agent.py on every task in tasks/ and print a pass/fail table.
# Use it once with your REAL models before the exam to see speed and success rate.
#
#   python smoke_test.py                                   # real Ollama, default models
#   python smoke_test.py --model qwen2.5-coder:1.5b --max-iters 6
#   python smoke_test.py --fake                            # no models needed (fake server)
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def make_test_image(path):
    """Synthetic image with red/green/blue shapes and a line, so every sample task has content."""
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    cv2.circle(img, (260, 90), 30, (0, 255, 0), -1)
    cv2.circle(img, (320, 220), 35, (0, 0, 255), -1)
    cv2.rectangle(img, (60, 190), (200, 260), (255, 0, 0), -1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(HERE / "tasks"))
    ap.add_argument("--docs", default=str(HERE / "docs"))
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--embed", default="nomic-embed-text")
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out", default="smoke_runs")
    ap.add_argument("--fake", action="store_true", help="start tests/fake_ollama_server.py and use it")
    args = ap.parse_args()

    env = dict(os.environ)
    srv = None
    if args.fake:
        sys.path.insert(0, str(HERE / "tests"))
        import fake_ollama_server
        srv = fake_ollama_server.start(11436)
        env["OLLAMA_HOST"] = "http://127.0.0.1:11436"

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    image = out / "input.png"
    make_test_image(image)
    db = out / ("chroma_db_fake" if args.fake else "chroma_db")

    rows = []
    for task_file in sorted(Path(args.tasks).glob("*.txt")):
        wd = out / task_file.stem
        cmd = [sys.executable, str(HERE / "rag_agent.py"), "--docs", args.docs, "--db", str(db),
               "--task-file", str(task_file), "--image", str(image), "--model", args.model,
               "--embed", args.embed, "--max-iters", str(args.max_iters),
               "--timeout", str(args.timeout), "--workdir", str(wd)]
        if args.vlm:
            cmd += ["--vlm", args.vlm]
        print(f"\n=== {task_file.name} ===", flush=True)
        t = time.time()
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        secs = time.time() - t
        (out / f"{task_file.stem}.log").write_text(p.stdout + p.stderr)
        iters = p.stdout.count(": asking the LLM")          # LLM calls, incl. ones that failed static checks
        status = "PASS" if p.returncode == 0 else "FAIL"
        print(f"{status} after {iters} attempt(s) in {secs:.0f}s  (log: {out / (task_file.stem + '.log')})")
        rows.append((task_file.stem, status, iters, secs))

    print("\n| task | result | attempts | seconds |\n| --- | --- | --- | --- |")
    for name, status, iters, secs in rows:
        print(f"| {name} | {status} | {iters} | {secs:.0f} |")
    passed = sum(r[1] == "PASS" for r in rows)
    print(f"\n{passed}/{len(rows)} tasks solved")
    if srv:
        srv.shutdown()
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
