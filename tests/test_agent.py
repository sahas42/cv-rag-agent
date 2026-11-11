# Offline tests: no Ollama and no real models needed.   Run:  python -m pytest -q
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rag_agent as ra  # noqa: E402

FENCE = "`" * 3


# ---------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("reply, expected", [
    (f"Here:\n{FENCE}python\nimport cv2\nprint(1)\n{FENCE}\nDone", "import cv2\nprint(1)"),
    (f"{FENCE}\nx = 1\n{FENCE}", "x = 1"),
    (f"<think>{FENCE}python\nbad\n{FENCE}</think>{FENCE}py\nok = 1\n{FENCE}", "ok = 1"),
    (f"{FENCE}python\nimport cv2\nprint(2)", "import cv2\nprint(2)"),          # unterminated fence
    ("import numpy as np\nprint(3)", "import numpy as np\nprint(3)"),           # no fence at all
    (f"{FENCE}python\nshort\n{FENCE}\n{FENCE}python\nmuch longer block\n{FENCE}", "much longer block"),
])
def test_extract_code(reply, expected):
    assert ra.extract_code(reply) == expected


def test_static_check_catches_problems():
    assert "SyntaxError" in ra.static_check("print((")
    assert "No module named 'torch_xyz_missing'" in ra.static_check("import torch_xyz_missing")
    assert "input()" in ra.static_check("x = input()")
    assert ra.static_check("import cv2, numpy as np\nprint(np.zeros(3))") is None


def test_strip_gui_keeps_indentation():
    code = "import cv2\nif True:\n    cv2.imshow('a', 1)\n    cv2.waitKey(0)\nx = 1\n"
    out = ra.strip_gui(code)
    assert "imshow" not in out.replace("# removed GUI call", "")
    compile(out, "t", "exec")                                   # still valid Python


@pytest.mark.parametrize("err, word", [
    ("AttributeError: 'NoneType' object has no attribute 'shape'", "imread"),
    ("ValueError: not enough values to unpack (expected 3, got 2) findContours", "findContours"),
    ("cv2.error: Overload resolution failed", "signature"),
    ("ModuleNotFoundError: No module named 'torch'", "not installed"),
    ("TimeoutError: script did not finish in 60s", "imshow"),
])
def test_hints(err, word):
    assert word in ra.hint_for(err)


def test_run_code_success_failure_timeout(tmp_path):
    ok, out, _ = ra.run_code("print('hi')", tmp_path, attempt=1)
    assert ok and out.strip() == "hi"
    ok, _, err = ra.run_code("1/0", tmp_path, attempt=2)
    assert not ok and "ZeroDivisionError" in ra.trim_error(err)
    ok, _, err = ra.run_code("while True: pass", tmp_path, timeout=2, attempt=3)
    assert not ok and "TimeoutError" in err


# ---------------------------------------------------------------- loop with a fake LLM object
def test_fix_loop_with_fake_llm(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    replies = [
        f"{FENCE}python\nprint('x'\n{FENCE}",                                          # syntax error
        f"{FENCE}python\nimport cv2, numpy as np\nimg = np.zeros((99, 99, 3), np.uint8)\n"
        f"e = cv2.Canny(img, 100)\n{FENCE}",                                           # wrong args
        f"{FENCE}python\nimport cv2, numpy as np\nimg = np.zeros((100, 100), np.uint8)\n"
        "cv2.rectangle(img, (20, 20), (80, 80), 255, -1)\ncv2.imshow('x', img)\n"
        f"e = cv2.Canny(img, 100, 200)\ncv2.imwrite('edges.png', e)\nprint(int((e > 0).sum()))\n{FENCE}",
    ]
    vs = ra.get_vectorstore(str(ROOT / "docs"), str(tmp_path / "db"), DeterministicFakeEmbedding(size=64))
    code = ra.solve("Canny edges", FakeListChatModel(responses=replies), vs,
                    workdir=str(tmp_path / "runs"), max_iters=4)
    assert code is not None
    assert (tmp_path / "runs" / "solution.py").exists()
    assert (tmp_path / "runs" / "edges.png").exists()


# ---------------------------------------------------------------- full CLI over HTTP (fake Ollama)
@pytest.fixture(scope="module")
def fake_ollama():
    import fake_ollama_server
    srv = fake_ollama_server.start(11437)
    yield "http://127.0.0.1:11437"
    srv.shutdown()


@pytest.mark.parametrize("task, out_file", [
    ("Detect edges in the image with Canny and save edges.png", "edges.png"),
    ("Count the objects in the image and save count.png", "count.png"),
    ("Make a mask of the red regions and save mask.png", "mask.png"),
])
def test_cli_end_to_end_fake_server(fake_ollama, tmp_path, task, out_file):
    env = dict(os.environ, OLLAMA_HOST=fake_ollama)
    p = subprocess.run([sys.executable, str(ROOT / "rag_agent.py"), "--docs", str(ROOT / "docs"),
                        "--db", str(tmp_path / "db"), "--task", task, "--workdir", str(tmp_path / "runs"),
                        "--vlm", "qwen2.5vl:3b"], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "iteration 2/" in p.stdout                       # first attempt failed, fix succeeded
    assert (tmp_path / "runs" / out_file).exists()
    assert (tmp_path / "runs" / "solution.py").exists()


def test_check_env_against_fake_server(fake_ollama):
    env = dict(os.environ, OLLAMA_HOST=fake_ollama)
    p = subprocess.run([sys.executable, str(ROOT / "check_env.py"), "--vlm", "qwen2.5vl:3b"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout
    assert "ALL GOOD" in p.stdout


# ---------------------------------------------------------------- every CV example in docs/ runs
def test_cv_examples_run_in_sequence(tmp_path):
    import cv2
    import numpy as np
    img = np.full((300, 400, 3), 40, np.uint8)
    cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
    cv2.circle(img, (250, 150), 30, (0, 255, 0), -1)
    cv2.rectangle(img, (180, 200), (330, 270), (255, 0, 0), -1)
    cv2.line(img, (10, 280), (390, 20), (255, 255, 255), 2)
    cv2.imwrite(str(tmp_path / "input.jpg"), img)
    ns, failures = {}, []
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for md in sorted((ROOT / "docs" / "examples").glob("*.md")):
            code = ra.extract_code(md.read_text())
            try:
                exec(compile(code, md.name, "exec"), ns)
            except Exception as e:                              # collect, report all at once
                failures.append(f"{md.name}: {type(e).__name__}: {e}")
    finally:
        os.chdir(cwd)
    assert not failures, "\n".join(failures)
