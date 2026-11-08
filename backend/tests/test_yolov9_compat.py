"""The pkg_resources stand-in the YOLOv9 launcher provides on setuptools 81+."""

from __future__ import annotations

import io

import pytest

from app.plugins.builtin import yolov9_compat


def test_stand_in_covers_what_yolov9_uses():
    pkg = yolov9_compat._pkg_resources_stand_in()
    assert pkg.parse_version("1.10") > pkg.parse_version("1.9")
    reqs = list(pkg.parse_requirements(io.StringIO("# comment\nnumpy>=1.0  # inline\n-r other.txt\n\npytest\n")))
    assert [f"{r.name}{r.specifier}" for r in reqs] == ["numpy>=1.0", "pytest"]
    pkg.require("pytest>=1")  # installed and new enough
    with pytest.raises(pkg.DistributionNotFound):
        pkg.require("no-such-package-visionloop")
    with pytest.raises(pkg.VersionConflict):
        pkg.require("pytest<1")
