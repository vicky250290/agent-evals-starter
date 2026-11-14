"""A minimal trace shape. Map your tracing tool's export onto it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass
class Trace:
    """One agent run: what went in, which tools it called, what came out."""

    id: str
    input: str
    output: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    success: bool | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_path(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.tool_calls)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        calls = [ToolCall(**c) for c in d.get("tool_calls", [])]
        return cls(**{**d, "tool_calls": calls})


def load_traces(path: str | Path) -> list[Trace]:
    """Load traces from a JSON Lines file (one trace object per line)."""
    with open(path, encoding="utf-8") as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]
