#!/usr/bin/env python3
# [FAKE-OLLAMA] A tiny Ollama-API-compatible server for rehearsing WITHOUT real models.
#
# It speaks the same HTTP API as `ollama serve` (/api/tags, /api/chat, /api/generate,
# /api/embed, /api/show, /api/version), so ChatOllama, OllamaEmbeddings and the `ollama`
# package all work against it unchanged. The "coder" answers with a deliberately BUGGY
# script first and a FIXED one once the prompt contains "It FAILED", which exercises the
# agent's run -> error -> fix loop. Embeddings are deterministic hashed bag-of-words vectors.
#
#   python tests/fake_ollama_server.py --port 11435
#   OLLAMA_HOST=http://127.0.0.1:11435 python rag_agent.py --docs docs --task "Detect edges ..."
import argparse
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["qwen2.5-coder:3b", "qwen2.5-coder:1.5b", "nomic-embed-text:latest",
          "qwen2.5vl:3b", "moondream:latest"]
DIM = 256
FENCE = "`" * 3

# ------------------------------------------------------------------ scripted code answers
SCENARIOS = {
    "edges": {
        "keys": ["edge", "canny"],
        "buggy": """import cv2, numpy as np
img = cv2.imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100)          # BUG: missing threshold2
cv2.imwrite('edges.png', edges)
""",
        "fixed": """import cv2, numpy as np, os
img = cv2.imread(IMG) if os.path.exists(IMG) else None
if img is None:
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.rectangle(img, (60, 60), (260, 180), (255, 255, 255), -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite('edges.png', edges)
print('saved edges.png', edges.shape, 'edge pixels:', int((edges > 0).sum()))
""",
    },
    "count": {
        "keys": ["count", "coin", "circle", "object"],
        "buggy": """import cv2, numpy as np
img = np.zeros((240, 320), np.uint8)
for c in [(60, 60), (160, 120), (260, 180)]:
    cv2.circle(img, c, 25, 255, -1)
_, cnts, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # BUG: OpenCV 3 API
print(len(cnts))
""",
        "fixed": """import cv2, numpy as np, os
img = cv2.imread(IMG) if os.path.exists(IMG) else None
if img is None:
    img = np.zeros((240, 320, 3), np.uint8)
    for c in [(60, 60), (160, 120), (260, 180)]:
        cv2.circle(img, c, 25, (255, 255, 255), -1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c for c in cnts if cv2.contourArea(c) > 50]
vis = img.copy()
cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
cv2.imwrite('count.png', vis)
print('objects:', len(cnts))
""",
    },
    "color": {
        "keys": ["red", "colour", "color", "mask", "hsv"],
        "buggy": """import cv2, numpy as np
img = cv2.imread('does_not_exist.jpg')
print(img.shape)                       # BUG: imread returned None
""",
        "fixed": """import cv2, numpy as np, os
img = cv2.imread(IMG) if os.path.exists(IMG) else None
if img is None:
    img = np.zeros((240, 320, 3), np.uint8)
    cv2.circle(img, (100, 120), 40, (0, 0, 255), -1)
    cv2.circle(img, (220, 120), 40, (0, 255, 0), -1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
cv2.imwrite('mask.png', mask)
print('red pixels:', cv2.countNonZero(mask))
""",
    },
    "default": {
        "keys": [],
        "buggy": """import cv2
print('unfinished'
""",
        "fixed": """import cv2, numpy as np, os
img = cv2.imread(IMG) if os.path.exists(IMG) else np.full((240, 320, 3), 127, np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('result.png', gray)
print('saved result.png', gray.shape)
""",
    },
}


def pick_scenario(text):
    task = text.split("TASK:", 1)[-1][:400].lower()
    for name, sc in SCENARIOS.items():
        if any(k in task for k in sc["keys"]):
            return sc
    return SCENARIOS["default"]


def coder_reply(prompt):
    if "TASK:" not in prompt:                            # e.g. check_env.py smoke chat
        return "print('hello from fake ollama')"
    sc = pick_scenario(prompt)
    m = re.search(r"Input image path: (.+)", prompt)
    img = repr(m.group(1).strip()) if m else "'input.jpg'"
    code = sc["fixed"] if "It FAILED" in prompt else sc["buggy"]
    return f"Here is the script:\n{FENCE}python\n{code.replace('IMG', img)}{FENCE}\n"


def reply_for(messages):
    if any(m.get("images") for m in messages):
        return "YES. The output image looks like a plausible result for the task."
    user = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return coder_reply(user[-1] if user else "")


def embed(text):
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ------------------------------------------------------------------ HTTP layer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # keep the console quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._json({"models": [{"name": m, "model": m, "modified_at": "2026-01-01T00:00:00Z",
                                    "size": 1, "digest": "fake",
                                    "details": {"format": "gguf", "family": "fake"}} for m in MODELS]})
        elif self.path.startswith("/api/version"):
            self._json({"version": "0.0.0-fake"})
        elif self.path in ("/", ""):
            self.send_response(200); self.end_headers(); self.wfile.write(b"Ollama is running")
        else:
            self._json({"error": "not found"}, 404)

    def _stream_or_single(self, body, key, text, extra):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        base = {"model": body.get("model", ""), "created_at": now}
        final = dict(base, done=True, done_reason="stop", total_duration=1000, load_duration=1,
                     prompt_eval_count=10, prompt_eval_duration=1, eval_count=len(text.split()),
                     eval_duration=1, **extra)
        if body.get("stream", True):                     # Ollama's default is streaming NDJSON
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in re.findall(r"\S*\s*", text):
                if not piece:
                    continue
                chunk = dict(base, done=False)
                chunk[key] = {"role": "assistant", "content": piece} if key == "message" else piece
                self.wfile.write((json.dumps(chunk) + "\n").encode())
            final[key] = {"role": "assistant", "content": ""} if key == "message" else ""
            self.wfile.write((json.dumps(final) + "\n").encode())
        else:
            final[key] = {"role": "assistant", "content": text} if key == "message" else text
            self._json(final)

    def do_POST(self):
        body = self._body()
        model = body.get("model", "")
        if self.path.startswith(("/api/chat", "/api/generate", "/api/embed", "/api/show")) and \
                model and model not in MODELS and model + ":latest" not in MODELS:
            return self._json({"error": f"model '{model}' not found"}, 404)
        if self.path.startswith("/api/chat"):
            self._stream_or_single(body, "message", reply_for(body.get("messages", [])), {})
        elif self.path.startswith("/api/generate"):
            msgs = [{"role": "user", "content": body.get("prompt", ""), "images": body.get("images")}]
            self._stream_or_single(body, "response", reply_for(msgs), {})
        elif self.path.startswith("/api/embed"):
            inp = body.get("input", "")
            texts = [inp] if isinstance(inp, str) else list(inp)
            self._json({"model": model, "embeddings": [embed(t) for t in texts]})
        elif self.path.startswith("/api/show"):
            self._json({"modelfile": "", "parameters": "", "template": "{{ .Prompt }}",
                        "details": {"format": "gguf", "family": "fake"}, "model_info": {},
                        "capabilities": ["completion", "vision"]})
        else:
            self._json({"error": "not found"}, 404)


def start(port=11435):
    """Start the server in a background thread; returns the server (call .shutdown() to stop)."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()
    print(f"fake Ollama on http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
