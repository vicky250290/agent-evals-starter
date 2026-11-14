"""Run: python examples/demo.py

Shows the workflow on a toy agent: flakiness first, then narrow graders,
then honest reliability numbers.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_evals import pass_at_k, pass_hat_k, wilson_interval  # noqa: E402
from agent_evals import graders as g  # noqa: E402
from agent_evals.flakiness import measure_flakiness, print_report  # noqa: E402
from toy_agent import REFUNDS, run_agent  # noqa: E402

TASKS = ["Refund order A123", "What is your return policy?", "Where is my parcel?"]

print("== 1. Flakiness: run each task 5 times ==")
print_report(measure_flakiness(run_agent, TASKS, n=5))

print("\n== 2. Graders on 10 refund runs ==")
n, ok = 10, 0
for _ in range(n):
    REFUNDS.clear()
    trace = run_agent("Refund order A123")
    results = [
        g.final_state("refund_state", REFUNDS, [("A123", 500)]),
        g.must_call(trace, "search_kb"),
        g.Result(
            "valid_customer_id",
            not any("@" in str(a.get("customer_id")) for a in g.args_of(trace, "get_orders")),
        ),
        g.step_budget(trace, 8),
        g.cost_budget(trace, 0.05),
    ]
    ok += g.passed(results)
    broken = ", ".join(r.name for r in g.failures(results)) or "-"
    print(f"{trace.id}: {'PASS' if g.passed(results) else 'FAIL'}  failed checks: {broken}")

print("\n== 3. Honest numbers ==")
lo, hi = wilson_interval(ok, n)
print(f"pass@1 = {ok}/{n} ({ok / n:.0%}), 95% CI {lo:.0%}-{hi:.0%}")
print(f"pass@3 = {pass_at_k(n, ok, 3):.0%}  (can it ever succeed?)")
print(f"pass^3 = {pass_hat_k(n, ok, 3):.0%}  (does it succeed every time?)")
