"""Common agent interface for MADKE-X.

Every agent is a callable ``agent(messages) -> AgentResult``.  The result makes
failures *explicit* (fix C7/C8): callers must check ``result.ok`` instead of
scoring an error string as if it were an answer.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AgentResult:
    ok: bool
    text: str = ""
    error: Optional[str] = None
    attempts: int = 1
    latency: float = 0.0
    usage: dict = field(default_factory=dict)   # prompt_tokens / completion_tokens when reported

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_value(value):
        """Adapt a legacy ``str`` return value into an :class:`AgentResult`."""
        if isinstance(value, AgentResult):
            return value
        if value is None:
            return AgentResult(ok=False, text="", error="agent returned None")
        text = str(value)
        if text == "Error":
            return AgentResult(ok=False, text="", error="legacy agent returned 'Error'")
        return AgentResult(ok=True, text=text)


class BaseAgent:
    """Minimal base class: subclasses implement :meth:`generate`."""

    def __init__(self, name="Agent"):
        self.agent_name = name

    def generate(self, messages):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, messages):
        return self.generate(messages)

    # kept for compatibility with the original torch-style ``forward`` API
    def forward(self, messages):
        return self.generate(messages)
