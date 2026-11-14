# Agent Evals Starter

A small, framework-neutral Python toolkit for answering one question: **is your AI agent actually working?**

Agents fail differently from normal software. A normal bug throws an exception; an agent bug returns a confident, well-formatted, wrong answer with HTTP 200: the right tool with the wrong ID, extra loops, a quietly skipped step. This repo gives you the pieces to catch that, in the order that works:

1. **Flakiness**: run each task several times. A single pass proves the agent *can* succeed, not that it *will*.
2. **Error analysis**: read 50–100 real traces and label the first failure in each, *before* writing any metric.
3. **Code-based graders**: narrow, named checks for the failures you found (final state, forbidden tools, argument validity, budgets).
4. **LLM-as-a-judge, calibrated**: one binary judge per fuzzy failure mode, measured against human labels (TPR/TNR).
5. **Honest metrics**: pass@k vs pass^k, confidence intervals, and which tasks flipped between versions.

Pure standard library, Python 3.10+. Works with LangGraph, the OpenAI Agents SDK, the Claude Agent SDK, or plain API calls: you only map your traces onto a small `Trace` dataclass.

## Try it (no API key needed)

```bash
git clone <this repo>
cd agent-evals-starter
python examples/demo.py
```

The demo runs a toy support agent that has three planted failure modes:

```text
== 1. Flakiness: run each task 5 times ==
'Refund order A123': 3/5 passed, 1 distinct tool paths  <-- flaky
'What is your return policy?': 4/5 passed, 2 distinct tool paths  <-- flaky
'Where is my parcel?': 5/5 passed, 1 distinct tool paths

== 2. Graders on 10 refund runs ==
t16: FAIL  failed checks: valid_customer_id
t17: PASS  failed checks: -
t18: FAIL  failed checks: called:search_kb
...

== 3. Honest numbers ==
pass@1 = 4/10 (40%), 95% CI 17%-69%
pass@3 = 83%  (can it ever succeed?)
pass^3 = 3%  (does it succeed every time?)
```

That last pair is the point: an agent can look capable (pass@3 = 83%) while being nowhere near reliable (pass^3 = 3%). Customer-facing agents need pass^k.

## Use it on your agent

**1. Map your traces.** Export from your tracing tool (Langfuse, LangSmith, OpenTelemetry, your own logs) into JSON Lines matching `Trace`:

```json
{"id": "t1", "input": "Refund order A123", "output": "Refunded A123.",
 "tool_calls": [{"name": "get_orders", "args": {"customer_id": "C42"}}],
 "success": true, "cost_usd": 0.004}
```

**2. Error analysis first.**

```python
from agent_evals import load_traces
from agent_evals.error_analysis import export_sample, count_categories

export_sample(load_traces("traces.jsonl"), "error_analysis.csv", n=60)
# Open the CSV. For each row, write a note on the FIRST failure you see,
# then group notes into 5-8 categories. Then:
print(count_categories("error_analysis.csv"))
# [('wrong customer looked up', 14), ('refund without policy check', 9), ...]
```

The top two or three categories are the evals to write.

**3. Write narrow graders** for those failures:

```python
from agent_evals import graders as g

results = [
    g.final_state("refund_state", refunds_in_db, [("A123", 500)]),  # strongest check
    g.not_called(trace, "delete_account"),
    g.step_budget(trace, 8),
    g.cost_budget(trace, 0.05),
]
print(g.passed(results), [r.name for r in g.failures(results)])
```

**4. Judge the fuzzy ones, then calibrate the judge.**

```python
from agent_evals.judge import make_judge, calibrate

judge = make_judge(my_llm_complete)          # any fn: prompt -> text, temperature 0
tpr, tnr = calibrate(judge, labelled)        # [(trace_text, human_pass), ...]
```

If TNR comes back at 60%, the judge misses 4 in 10 real failures: fix the prompt before trusting it.

**5. Report honestly.**

```python
from agent_evals import pass_hat_k, wilson_interval
from agent_evals.metrics import flips

print(pass_hat_k(n=5, c=4, k=5))       # reliability across 5 tries
print(wilson_interval(41, 50))         # on 50 tasks, about +/-10 points
print(flips(old_results, new_results)) # read the traces of tasks that flipped
```

## Run the tests

```bash
python -m unittest discover -s tests -t . -v
```

## Learn the method

This repo is the companion to **[Is Your AI Agent Actually Working? Evals, Observability & Debugging for Production Agents](https://www.techbrosebooks.co.in/books/is-your-ai-agent-actually-working-evals-observability-debugging-for-production-agents?utm_source=github&utm_medium=referral&utm_campaign=agent-evals-starter)**, which covers tracing with OpenTelemetry, building eval datasets, judge biases and fixes, evals in CI and production, and an end-to-end fix of a support-triage agent.

Free article: [How to Test an AI Agent: Start With Error Analysis, Not Metrics](https://www.techbrosebooks.co.in/blog/how-to-test-an-ai-agent?utm_source=github&utm_medium=referral&utm_campaign=agent-evals-starter)

## License

[MIT](LICENSE)
