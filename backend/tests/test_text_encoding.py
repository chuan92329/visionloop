"""Every text file the platform reads or writes says UTF-8.

Without an explicit encoding Python uses the Windows code page (cp950 on Traditional Chinese
Windows), so a data.yaml with Chinese class names could not be read back: YOLOv9 training failed
whenever the platform was started without UTF-8 mode.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted([*(ROOT / "backend" / "app").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])


def text_io_without_encoding(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        keywords = {k.arg for k in node.keywords}
        if "encoding" in keywords:
            continue
        if name in ("read_text", "write_text"):
            found.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}()")
        elif name == "open":
            owner = getattr(func, "value", None)
            if isinstance(owner, ast.Name) and owner.id.lower().endswith("image"):
                continue  # PIL Image.open: binary
            # open(file, mode) for the builtin, path.open(mode) for a Path
            positional = node.args[0:1] if isinstance(func, ast.Attribute) else node.args[1:2]
            modes = [a.value for a in positional if isinstance(a, ast.Constant)]
            modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
            if not any("b" in str(m) for m in modes):
                found.append(f"{path.relative_to(ROOT)}:{node.lineno} open()")
    return found


def test_text_files_are_read_and_written_as_utf8():
    problems = [p for f in SOURCES for p in text_io_without_encoding(f)]
    assert problems == [], "add encoding=\"utf-8\":\n" + "\n".join(problems)
