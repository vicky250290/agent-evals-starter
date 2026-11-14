"""Small, framework-neutral helpers for testing LLM agents.

Start with error analysis (read real traces), then add narrow graders,
then measure reliability with pass^k. See README.md.
"""

from agent_evals.trace import ToolCall, Trace, load_traces
from agent_evals.graders import Result, args_of, called, passed
from agent_evals.metrics import pass_at_k, pass_hat_k, wilson_interval

__all__ = [
    "ToolCall",
    "Trace",
    "load_traces",
    "Result",
    "args_of",
    "called",
    "passed",
    "pass_at_k",
    "pass_hat_k",
    "wilson_interval",
]
