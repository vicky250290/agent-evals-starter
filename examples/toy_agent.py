"""A fake support agent with realistic failure modes, so the demo runs
without an API key. Replace run_agent with a call to your real agent that
returns a Trace."""

from __future__ import annotations

import random

from agent_evals import ToolCall, Trace

_rng = random.Random(7)
_counter = 0

# Shared "database" the toy agent writes to; graders check it.
REFUNDS: list[tuple[str, int]] = []


def run_agent(task: str) -> Trace:
    global _counter
    _counter += 1
    calls: list[ToolCall] = []
    output = ""

    if task.startswith("Refund order"):
        order = task.split()[-1]
        # Failure mode 1: sometimes looks the customer up by email, not ID.
        if _rng.random() < 0.3:
            calls.append(ToolCall("get_orders", {"customer_id": "a@b.com"}))
        else:
            calls.append(ToolCall("get_orders", {"customer_id": "C42"}))
        # Failure mode 2: sometimes skips the policy check.
        if _rng.random() < 0.75:
            calls.append(ToolCall("search_kb", {"query": "refund policy"}))
        calls.append(ToolCall("issue_refund", {"order_id": order, "amount": 500}))
        REFUNDS.append((order, 500))
        output = f"Refunded {order}."
    elif task.startswith("Where is my parcel"):
        calls.append(ToolCall("track_parcel", {"order_id": "A123"}))
        output = "It arrives tomorrow."
    else:
        # Failure mode 3: answers from memory instead of the knowledge base.
        if _rng.random() < 0.5:
            calls.append(ToolCall("search_kb", {"query": task}))
        output = "Returns are accepted within 30 days."

    return Trace(
        id=f"t{_counter}",
        input=task,
        output=output,
        tool_calls=calls,
        success=bool(calls) and "a@b.com" not in str(calls),
        cost_usd=0.004 * len(calls),
    )
