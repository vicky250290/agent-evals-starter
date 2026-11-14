"""Run each task several times to see how consistent the agent really is.

A single pass proves the agent *can* succeed, not that it *will*. If a task
passes 3 of 5 times with four different tool paths, you have found your
first reliability problem without writing a single grader.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from agent_evals.trace import Trace


@dataclass
class TaskFlakiness:
    task: str
    runs: int
    passes: int
    distinct_paths: int
    paths: Counter

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.runs or self.distinct_paths > 1


def measure_flakiness(
    run_agent: Callable[[str], Trace], tasks: list[str], n: int = 5
) -> list[TaskFlakiness]:
    report = []
    for task in tasks:
        traces = [run_agent(task) for _ in range(n)]
        paths = Counter(t.tool_path for t in traces)
        report.append(
            TaskFlakiness(
                task=task,
                runs=n,
                passes=sum(bool(t.success) for t in traces),
                distinct_paths=len(paths),
                paths=paths,
            )
        )
    return report


def print_report(report: list[TaskFlakiness]) -> None:
    for r in report:
        flag = "  <-- flaky" if r.flaky else ""
        print(
            f"{r.task!r}: {r.passes}/{r.runs} passed, "
            f"{r.distinct_paths} distinct tool paths{flag}"
        )
