"""Train / valid / test split.

Design goals
------------
* **Only human-reviewed images** enter a split.
* **Groups are never torn apart** – frames of the same video, crops of the
  same scene, etc. share a ``group_key`` and always land in the same split,
  otherwise near-duplicate images leak from train into test and the test
  score is meaningless.
* **Stratified** – every class should appear in every split in roughly the
  requested ratio, including rare classes.  We use a greedy variant of
  *iterative stratification* (Sechidis et al., 2011) that works on groups:
  groups containing the rarest classes are placed first, each into the split
  where it reduces the deviation from the target image and box counts the
  most (a chi-square style cost, so small splits are not starved).
* **Deterministic** for a given seed.
* **Incremental by default** – existing assignments are kept, so adding new
  data does not silently move images between splits.
* **Locked test set** – once locked, test images never move and new images
  only go to train / valid, so models trained at different times are
  evaluated on exactly the same test images.

The core algorithm is a pure function (``plan_split``) so it can be unit
tested without a database.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SPLITS = ("train", "valid", "test")


@dataclass
class SplitItem:
    image_id: int
    group_key: str | None
    current_split: str | None
    reviewed: bool
    class_counts: Counter = field(default_factory=Counter)


@dataclass
class SplitPlan:
    assignments: dict[int, str | None]  # image_id -> new split (None = clear)
    warnings: list[str]


def _group_of(item: SplitItem) -> str:
    return item.group_key if item.group_key else f"__image_{item.image_id}"


def plan_split(
    items: list[SplitItem],
    ratios: dict[str, float],
    *,
    seed: int = 42,
    mode: str = "incremental",
    stratify: bool = True,
    test_locked: bool = False,
    class_names: dict[int, str] | None = None,
) -> SplitPlan:
    warnings: list[str] = []
    assignments: dict[int, str | None] = {}

    allowed = [s for s in SPLITS if ratios.get(s, 0) > 0]
    if test_locked:
        allowed = [s for s in allowed if s != "test"]
    if not allowed:
        raise ValueError("沒有可用的切分比例")

    # ---- 1. decide which reviewed images are fixed
    eligible = [it for it in items if it.reviewed]
    fixed: dict[int, str] = {}
    for it in eligible:
        if it.current_split is None:
            continue
        if test_locked and it.current_split == "test":
            fixed[it.image_id] = "test"
        elif mode == "incremental":
            fixed[it.image_id] = it.current_split

    # In full mode unreviewed images are taken out of their split (locked test excepted)
    if mode == "full":
        for it in items:
            if not it.reviewed and it.current_split and not (test_locked and it.current_split == "test"):
                assignments[it.image_id] = None

    # ---- 2. build groups
    groups: dict[str, list[SplitItem]] = defaultdict(list)
    for it in eligible:
        groups[_group_of(it)].append(it)

    free_groups: list[tuple[str, list[SplitItem]]] = []
    for key, members in groups.items():
        fixed_splits = Counter(fixed[m.image_id] for m in members if m.image_id in fixed)
        unassigned = [m for m in members if m.image_id not in fixed]
        if not unassigned:
            continue
        if fixed_splits:
            target, _ = fixed_splits.most_common(1)[0]
            if len(fixed_splits) > 1:
                warnings.append(f"群組「{key}」的圖片目前分散在多個集合：{dict(fixed_splits)}")
            if test_locked and target == "test":
                # cannot grow a locked test set; keep the group out of train/valid as well
                warnings.append(f"群組「{key}」已在鎖定的 test 中，新加入的 {len(unassigned)} 張圖不會被分配")
                continue
            for m in unassigned:
                assignments[m.image_id] = target
                fixed[m.image_id] = target
        else:
            free_groups.append((key, unassigned))

    # ---- 3. targets
    total_imgs = len(eligible)
    total_cls: Counter = Counter()
    for it in eligible:
        total_cls.update(it.class_counts)

    current_imgs: Counter = Counter()
    current_cls: dict[str, Counter] = {s: Counter() for s in SPLITS}
    by_id = {it.image_id: it for it in eligible}
    for img_id, s in fixed.items():
        current_imgs[s] += 1
        current_cls[s].update(by_id[img_id].class_counts)

    if test_locked:
        # test is frozen: distribute the rest between train/valid with renormalised ratios
        n_test = current_imgs["test"]
        rest = sum(ratios[s] for s in allowed)
        desired_imgs = {s: (total_imgs - n_test) * ratios[s] / rest for s in allowed}
        desired_cls = {
            s: {c: (n - current_cls["test"][c]) * ratios[s] / rest for c, n in total_cls.items()}
            for s in allowed
        }
    else:
        desired_imgs = {s: total_imgs * ratios[s] for s in allowed}
        desired_cls = {s: {c: n * ratios[s] for c, n in total_cls.items()} for s in allowed}

    # ---- 4. order groups: shuffled (seeded), then rarest class first
    rng = random.Random(seed)
    free_groups.sort(key=lambda g: g[0])  # make the shuffle independent of DB order
    rng.shuffle(free_groups)

    def group_counts(members: list[SplitItem]) -> Counter:
        c: Counter = Counter()
        for m in members:
            c.update(m.class_counts)
        return c

    prepared = [(key, members, group_counts(members)) for key, members in free_groups]
    # rarest class first (iterative stratification); among equals, big groups first,
    # because they are the hardest to fit (like packing large items first)
    prepared.sort(
        key=lambda g: (
            -max((1.0 / total_cls[c] for c in g[2]), default=0.0) if stratify else 0.0,
            -len(g[1]),
        )
    )  # stable sort keeps the shuffled order among equals

    # balance image counts and box counts on a comparable scale
    total_boxes = sum(total_cls.values())
    cls_weight = (total_imgs / total_boxes) if total_boxes else 0.0

    def delta_cost(current: float, add: float, desired: float) -> float:
        """Change of (current - desired)^2 / desired when ``add`` is added – a chi-square
        style penalty, so missing the target of a small split hurts as much as a big one."""
        d = max(desired, 0.5)
        return ((current + add - d) ** 2 - (current - d) ** 2) / d

    # ---- 5. greedy assignment: put each group where it reduces the deviation most
    for _key, members, gcls in prepared:
        best_split, best_score = None, None
        n = len(members)
        for s in allowed:
            cost = delta_cost(current_imgs[s], n, desired_imgs[s])
            if stratify and gcls:
                cost += cls_weight * sum(
                    delta_cost(current_cls[s][c], k, desired_cls[s][c]) for c, k in gcls.items()
                )
            # lower cost wins; ties go to the larger split, then the canonical order
            cand = (-round(cost, 9), ratios[s], -SPLITS.index(s))
            if best_score is None or cand > best_score:
                best_split, best_score = s, cand
        assert best_split is not None
        for m in members:
            assignments[m.image_id] = best_split
            current_imgs[best_split] += 1
            current_cls[best_split].update(m.class_counts)

    # ---- 6. warnings about the result
    if total_imgs and total_imgs < 10:
        warnings.append(f"只有 {total_imgs} 張已確認的圖片，切分結果參考價值有限")
    for s in allowed:
        if desired_imgs[s] >= 5:
            dev = abs(current_imgs[s] - desired_imgs[s]) / max(total_imgs, 1)
            if dev > 0.1:
                warnings.append(
                    f"{s} 實際 {current_imgs[s]} 張，目標約 {desired_imgs[s]:.0f} 張；"
                    "通常是因為有很大的群組無法拆開"
                )
    final_splits = list(allowed) + (["test"] if test_locked and current_imgs["test"] else [])
    for c, n in total_cls.items():
        missing = [s for s in final_splits if current_cls[s][c] == 0]
        if missing and n >= len(final_splits):
            name = (class_names or {}).get(c, f"id {c}")
            warnings.append(f"類別「{name}」在 {', '.join(missing)} 中沒有任何標註")

    return SplitPlan(assignments=assignments, warnings=warnings)
