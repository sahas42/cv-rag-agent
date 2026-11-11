#!/usr/bin/env python3
# [CORPUS-BUILD] Dump the docstrings of the INSTALLED libraries into Markdown files for RAG.
# Works fully offline and always matches the versions on this machine.
#
#   python build_corpus.py                 # writes docs/api/*.md
#   python build_corpus.py docs/api cv2 numpy skimage.filters   # choose modules
import importlib
import inspect
import re
import sys
from pathlib import Path

DEFAULT_TARGETS = [
    "cv2", "numpy",
    "skimage.filters", "skimage.feature", "skimage.measure", "skimage.morphology",
    "skimage.segmentation", "skimage.transform", "skimage.exposure", "skimage.color",
    "skimage.metrics", "PIL.Image", "PIL.ImageFilter", "PIL.ImageOps",
]


def entries(modname):
    mod = importlib.import_module(modname)
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if not callable(obj):
            continue
        doc = inspect.getdoc(obj) or ""
        if not doc.strip():
            continue
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = ""                          # cv2 builtins: signature is inside the docstring
        yield name, sig, doc


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/api")
    targets = sys.argv[2:] or DEFAULT_TARGETS
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for modname in targets:
        try:
            items = list(entries(modname))
        except Exception as e:                # module not installed: skip it
            print(f"skip {modname}: {e}")
            continue
        with open(out / f"{modname.replace('.', '_')}.md", "w", encoding="utf-8") as f:
            f.write(f"# {modname} API reference\n\n")
            for name, sig, doc in items:
                doc = re.sub(r"^\.\s{0,3}", "", doc, flags=re.M)   # strip cv2's leading '.   '
                doc = re.sub(r"\n{3,}", "\n\n", doc)[:4000]          # cap huge numpy docstrings
                f.write(f"## {modname}.{name}{sig}\n\n{doc}\n\n")
        print(f"{modname}: {len(items)} entries")
        total += len(items)
    print(f"total {total} entries -> {out}/")


if __name__ == "__main__":
    main()
