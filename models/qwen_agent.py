"""Qwen agent served through any OpenAI-compatible server (vLLM / Ollama / llama.cpp).

Thin wrapper over :class:`models.openai_compat_agent.OpenAICompatAgent`; uses
``--base_url``, ``--api_key`` and ``--model`` and returns ``AgentResult``.
"""

from models.openai_compat_agent import OpenAICompatAgent


class Qwen_Agent(OpenAICompatAgent):
    def __init__(self, args, name):
        super().__init__(args, name)
