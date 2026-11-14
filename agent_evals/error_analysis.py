"""Error analysis: read traces before you write any metric.

1. Sample 50-100 real traces into a CSV (export_sample).
2. Open coding: for each, write a free-form note on the FIRST failure you see.
3. Axial coding: group the notes into 5-8 specific failure categories.
4. Count them (count_categories). The top two or three are the evals to write.

Stop when new traces stop revealing new failure modes (usually ~100).
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from agent_evals.trace import Trace

COLUMNS = [
    "trace_id",
    "input",
    "tool_calls",
    "final_output",
    "pass",
    "first_failure_note",
    "category",
]


def export_sample(
    traces: list[Trace], path: str | Path, n: int = 60, seed: int | None = None
) -> int:
    """Write a random sample to CSV; you fill the last three columns by hand."""
    rng = random.Random(seed)
    sample = rng.sample(traces, min(n, len(traces)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for t in sample:
            w.writerow(
                [t.id, t.input, " > ".join(t.tool_path), t.output, "", "", ""]
            )
    return len(sample)


def count_categories(path: str | Path) -> list[tuple[str, int]]:
    """Tally the hand-labelled categories, most common first."""
    with open(path, newline="", encoding="utf-8") as f:
        cats = [
            row["category"].strip()
            for row in csv.DictReader(f)
            if row.get("category", "").strip()
        ]
    return Counter(cats).most_common()
