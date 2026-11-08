"""Launcher for the YOLOv9 repository's scripts: ``python yolov9_compat.py train_dual.py ...``.

The repo was written for NumPy 1.x and calls functions NumPy 2.x removed
(``np.trapz`` in the mAP computation, ``np.float``).  Instead of editing its
(GPL) code or pinning an old NumPy for the whole models environment, this
restores those names and then runs the requested script as ``__main__``.

``segment/predict.py`` also has a bug of its own: it uses the model's auxiliary
output tuple as if it were the mask prototypes, so it crashes on the first
detection (``segment/val.py`` correctly takes the last element).  That line is
corrected in memory; the file on disk stays untouched.

Runs in the models environment, outside the platform; standard library only
besides NumPy.
"""

import re
import sys
from pathlib import Path

# module level, so DataLoader worker processes (spawned on Windows) get the shims too
try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    if not hasattr(np, "float"):
        np.float = float

SEG_PREDICT_BUG = re.compile(r"^( *)pred, proto = model\(im, augment=augment, visualize=visualize\)\[:2\]\n", re.M)
SEG_PREDICT_FIX = r"\g<0>\1proto = proto[-1]\n"

if __name__ == "__main__":
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path[0] = str(script.parent)  # as if the script itself had been started
    source = script.read_text(encoding="utf-8")
    if script.parent.name == "segment" and script.name == "predict.py":
        source = SEG_PREDICT_BUG.sub(SEG_PREDICT_FIX, source)
    exec(compile(source, str(script), "exec"), {"__name__": "__main__", "__file__": str(script)})
