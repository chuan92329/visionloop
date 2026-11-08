"""The probe learns from an object's outline when it has one, and from its box otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from app.db import SessionLocal
from app.models import Project
from app.plugins.builtin.probe_helper import object_areas, patch_targets, polygon_coverage
from app.services.jobs import _collect_reviewed

from .conftest import upload

# a thin bar from the top-left to the bottom-right corner of a 100 x 100 image
DIAGONAL = [[0, 0], [15, 0], [100, 85], [100, 100], [85, 100], [0, 15]]
DIAGONAL_BOX = [0, 0.0, 0.0, 100.0, 100.0]
GRID = 10  # patches per side; one patch is 10 x 10 pixels


def test_box_teaches_its_corners_as_object_but_polygon_does_not():
    box = patch_targets([DIAGONAL_BOX], 100, 100, GRID, GRID, 1)
    outline = patch_targets([DIAGONAL_BOX], 100, 100, GRID, GRID, 1, [DIAGONAL])
    # (row 0, last column) is the empty corner of the bounding box
    assert box[0, GRID - 1, 0] == pytest.approx(1)
    assert outline[0, GRID - 1, 0] == pytest.approx(0)
    assert outline[0, GRID - 1, 1] == pytest.approx(1)  # background
    # both agree on the patch in the middle of the bar
    assert outline[5, 5, 0] == pytest.approx(1)
    assert box[5, 5, 0] == pytest.approx(1)


def test_coverage_is_the_fraction_of_the_patch():
    half = [[0, 0], [5, 0], [5, 10], [0, 10]]  # the left half of the first patch
    cover = polygon_coverage(half, 100, 100, GRID, GRID)
    assert cover[0, 0] == pytest.approx(0.5, abs=0.1)
    assert cover[0, 1] == pytest.approx(0, abs=0.1)
    assert cover[1, 0] == pytest.approx(0, abs=0.1)


@pytest.mark.parametrize(
    "outline, area",
    [
        ([[31, 42], [54, 42], [54, 59], [31, 59]], 391),
        ([[10, 10], [70, 10], [10, 70]], 1800),
        ([[0, 0], [4, 0], [100, 96], [100, 100], [96, 100], [0, 4]], 784),  # thin: edges are most of its area
    ],
)
def test_coverage_adds_up_to_the_polygon_area(outline, area):
    # 100 pixels per patch.  The method converges on the true area; the default resolution stays close
    assert polygon_coverage(outline, 100, 100, GRID, GRID, supersample=64).sum() * 100 == pytest.approx(area, rel=0.03)
    assert polygon_coverage(outline, 100, 100, GRID, GRID).sum() * 100 == pytest.approx(area, rel=0.1)


def test_coverage_ignores_what_lies_outside_the_image():
    outline = [[-50, -50], [50, -50], [50, 50], [-50, 50]]
    cover = polygon_coverage(outline, 100, 100, GRID, GRID)
    assert cover.sum() * 100 == pytest.approx(2500, rel=0.01)
    assert cover[9, 9] == 0


def test_box_targets_stay_exact_without_polygons():
    boxes = [[0, 0.0, 0.0, 25.0, 10.0]]
    t = patch_targets(boxes, 100, 100, GRID, GRID, 1)
    assert t[0, 2, 0] == pytest.approx(0.5)
    assert np.array_equal(t, patch_targets(boxes, 100, 100, GRID, GRID, 1, [None]))
    assert np.array_equal(t, patch_targets(boxes, 100, 100, GRID, GRID, 1, []))


def test_mixed_objects_and_overlaps_stay_valid_probabilities():
    boxes = [DIAGONAL_BOX, [1, 0.0, 0.0, 100.0, 100.0]]  # the box of class 1 overlaps the outline of class 0
    t = patch_targets(boxes, 100, 100, GRID, GRID, 2, [DIAGONAL, None])
    assert t.sum(-1) == pytest.approx(np.ones((GRID, GRID)))
    assert (t >= 0).all()
    assert t[5, 5, 0] == pytest.approx(0.5)  # shared between the two classes
    assert t[0, GRID - 1, 1] == pytest.approx(1)  # only the box of class 1 reaches this corner


def test_object_areas_use_the_outline_where_there_is_one():
    triangle = [[0, 0], [10, 0], [0, 10]]
    samples = [
        {"boxes": [[0, 0, 0, 10, 10], [0, 0, 0, 4, 5]], "polygons": [triangle, None]},
        {"boxes": [[0, 0, 0, 2, 3]]},  # a spec without polygons
    ]
    assert object_areas(samples) == pytest.approx([50, 20, 6])


def test_confirmed_polygons_reach_the_probe(client, project):
    pid = project["id"]
    class_id = project["classes"][0]["id"]
    created = upload(client, pid, ["a.jpg"])["created"]
    outline = [[10, 10], [60, 10], [10, 60]]
    r = client.put(
        f"/api/images/{created[0]['id']}/annotations",
        json={
            "boxes": [
                {"class_id": class_id, "x": 10, "y": 10, "w": 50, "h": 50, "polygon": outline},
                {"class_id": class_id, "x": 100, "y": 10, "w": 30, "h": 30},
            ],
            "mark_reviewed": True,
        },
    )
    assert r.status_code == 200, r.text
    with SessionLocal() as db:
        reviewed = _collect_reviewed(db, db.get(Project, pid), 10)
    assert len(reviewed) == 1
    assert len(reviewed[0].boxes) == len(reviewed[0].polygons) == 2
    by_size = sorted(zip(reviewed[0].boxes, reviewed[0].polygons), key=lambda b: -b[0][3])
    assert by_size[0][1] == outline
    assert by_size[1][1] is None
