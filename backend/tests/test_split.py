from collections import Counter

import pytest

from app.services.split import SplitItem, plan_split

RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}


def items_from(spec):
    """spec: list of (group_key, {class: n})"""
    return [
        SplitItem(image_id=i, group_key=g, current_split=None, reviewed=True, class_counts=Counter(c))
        for i, (g, c) in enumerate(spec, start=1)
    ]


def apply(items, plan):
    for it in items:
        if it.image_id in plan.assignments:
            it.current_split = plan.assignments[it.image_id]


def test_ratios_roughly_respected():
    items = items_from([(None, {i % 3: 1}) for i in range(1000)])
    plan = plan_split(items, RATIOS, seed=1)
    n = Counter(plan.assignments.values())
    assert abs(n["train"] - 700) <= 10
    assert abs(n["valid"] - 200) <= 10
    assert abs(n["test"] - 100) <= 10


def test_rare_class_present_in_every_split():
    spec = [(None, {0: 1}) for _ in range(300)] + [(None, {1: 1}) for _ in range(6)]
    items = items_from(spec)
    plan = plan_split(items, RATIOS, seed=3)
    rare = Counter(plan.assignments[it.image_id] for it in items if 1 in it.class_counts)
    assert set(rare) == {"train", "valid", "test"}


def test_groups_never_split():
    spec = []
    for g in range(40):
        spec += [(f"video{g}", {g % 4: 2}) for _ in range(5)]
    items = items_from(spec)
    plan = plan_split(items, RATIOS, seed=0)
    by_group = {}
    for it in items:
        by_group.setdefault(it.group_key, set()).add(plan.assignments[it.image_id])
    assert all(len(s) == 1 for s in by_group.values())
    assert set(plan.assignments.values()) == {"train", "valid", "test"}


def test_deterministic_for_same_seed_and_independent_of_order():
    spec = [(None, {i % 5: 1 + i % 2}) for i in range(200)]
    a = plan_split(items_from(spec), RATIOS, seed=7).assignments
    b = plan_split(list(reversed(items_from(spec))), RATIOS, seed=7).assignments
    assert a == b
    c = plan_split(items_from(spec), RATIOS, seed=8).assignments
    assert a != c


def test_unreviewed_images_are_ignored():
    items = items_from([(None, {0: 1}) for _ in range(10)])
    items[0].reviewed = False
    plan = plan_split(items, RATIOS)
    assert items[0].image_id not in plan.assignments
    assert len(plan.assignments) == 9


def test_incremental_keeps_existing_assignments():
    items = items_from([(None, {i % 2: 1}) for i in range(100)])
    first = plan_split(items, RATIOS, seed=1)
    apply(items, first)
    new = items_from([(None, {0: 1}) for _ in range(20)])
    for k, it in enumerate(new):
        it.image_id = 1000 + k
    second = plan_split(items + new, RATIOS, seed=1, mode="incremental")
    assert set(second.assignments) == {it.image_id for it in new}


def test_locked_test_set_is_frozen():
    items = items_from([(None, {0: 1}) for _ in range(100)])
    apply(items, plan_split(items, RATIOS, seed=1))
    test_ids = {it.image_id for it in items if it.current_split == "test"}
    assert test_ids
    new = items_from([(None, {0: 1}) for _ in range(50)])
    for k, it in enumerate(new):
        it.image_id = 1000 + k
    plan = plan_split(items + new, RATIOS, seed=2, mode="full", test_locked=True)
    assert not (set(plan.assignments) & test_ids)  # test images untouched
    assert "test" not in plan.assignments.values()  # nothing new enters test
    train_valid = Counter(plan.assignments.values())
    # 140 non-test images split 70:20
    assert abs(train_valid["train"] / (train_valid["train"] + train_valid["valid"]) - 0.7 / 0.9) < 0.05


def test_new_member_of_existing_group_follows_its_group():
    items = items_from([("cam1", {0: 1})] * 3 + [(None, {1: 1}) for _ in range(30)])
    apply(items, plan_split(items, RATIOS, seed=1))
    cam_split = items[0].current_split
    extra = SplitItem(image_id=999, group_key="cam1", current_split=None, reviewed=True, class_counts=Counter({0: 1}))
    plan = plan_split(items + [extra], RATIOS, seed=1)
    assert plan.assignments[999] == cam_split


def test_zero_ratio_split_gets_nothing():
    items = items_from([(None, {0: 1}) for _ in range(50)])
    plan = plan_split(items, {"train": 0.8, "valid": 0.2, "test": 0.0})
    assert "test" not in plan.assignments.values()


def test_invalid_ratios():
    with pytest.raises(ValueError):
        plan_split(items_from([(None, {})]), {"train": 0, "valid": 0, "test": 0})


def test_few_large_groups_minimise_deviation():
    # 19 images in groups of 8, 8 and 3 -> the best achievable is 16 / 3 / 0, not 8 / 8 / 3
    spec = [("a", {0: 1})] * 8 + [("b", {0: 1})] * 8 + [("c", {0: 1})] * 3
    plan = plan_split(items_from(spec), RATIOS, seed=0)
    assert Counter(plan.assignments.values()) == Counter({"train": 16, "valid": 3})
