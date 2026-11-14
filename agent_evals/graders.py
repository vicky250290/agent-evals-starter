"""Code-based graders: fast, cheap and deterministic.

Each check returns a named Result, so a failing trial tells you which rule
broke. Final-state checks (what actually changed in the database) are the
strongest graders for agents that take actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from agent_evals.trace import Trace


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""


def called(trace: Trace, tool: str) -> bool:
    return any(c.name == tool for c in trace.tool_calls)


def args_of(trace: Trace, tool: str) -> list[dict[str, Any]]:
    return [c.args for c in trace.tool_calls if c.name == tool]


def final_state(name: str, actual: Iterable, expected: Iterable) -> Result:
    """Compare what the agent actually changed with what it should have."""
    a, e = sorted(actual), sorted(expected)
    return Result(name, a == e, f"actual={a} expected={e}")


def not_called(trace: Trace, tool: str) -> Result:
    return Result(f"not_called:{tool}", not called(trace, tool))


def must_call(trace: Trace, tool: str) -> Result:
    return Result(f"called:{tool}", called(trace, tool))


def step_budget(trace: Trace, max_steps: int) -> Result:
    n = len(trace.tool_calls)
    return Result("step_budget", n <= max_steps, f"{n} steps (max {max_steps})")


def cost_budget(trace: Trace, max_usd: float) -> Result:
    return Result(
        "cost_budget",
        trace.cost_usd <= max_usd,
        f"${trace.cost_usd:.3f} (max ${max_usd:.3f})",
    )


def passed(results: list[Result]) -> bool:
    return all(r.passed for r in results)


def failures(results: list[Result]) -> list[Result]:
    return [r for r in results if not r.passed]
