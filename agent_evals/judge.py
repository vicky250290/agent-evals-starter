"""LLM-as-a-judge, done properly: one narrow, binary judge per failure mode,
calibrated against human labels like a classifier.

Bring your own model call: `complete` is any function that takes a prompt
string and returns the model's text (use temperature 0). Prefer a judge
from a different provider than the agent to avoid self-enhancement bias.
"""

from __future__ import annotations

from typing import Callable

# An example for one failure mode found in error analysis: policy claims
# made from memory instead of the knowledge base.
JUDGE_PROMPT = """You are grading one behaviour of a support agent.
CRITERION: Every claim about refund or return policy must come from a
search_kb tool result visible in the trace. Claims from memory FAIL.
Length is not a virtue. Ignore tone and style.

<trace>{trace}</trace>

First write REASONING (2-3 sentences citing trace evidence).
Then write VERDICT: PASS or VERDICT: FAIL."""


def make_judge(
    complete: Callable[[str], str], prompt: str = JUDGE_PROMPT
) -> Callable[[str], bool]:
    """Return judge(trace_text) -> True for PASS."""

    def judge(trace_text: str) -> bool:
        out = complete(prompt.format(trace=trace_text))
        return "VERDICT: PASS" in out

    return judge


def calibrate(
    judge: Callable[[str], bool], labelled: list[tuple[str, bool]]
) -> tuple[float, float]:
    """Measure the judge against human labels: [(trace_text, human_pass)].

    Returns (TPR, TNR). TNR is the one to watch: at 60% the judge misses
    4 in 10 real failures. Fix the prompt before trusting its numbers, and
    re-calibrate whenever the prompt or judge model changes.
    """
    tp = fn = tn = fp = 0
    for text, human in labelled:
        j = judge(text)
        if human and j:
            tp += 1
        elif human and not j:
            fn += 1
        elif not human and not j:
            tn += 1
        else:
            fp += 1
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)
