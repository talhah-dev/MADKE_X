"""ChatGPT (Azure OpenAI) agent -- thin wrapper over the failure-aware client.

Kept for backward compatibility with the original MADKE command line
(``--endpoint``, ``--deployment_id``, ``--api_key``).  The agent now returns an
:class:`models.base.AgentResult` ``(ok, text, error, attempts)`` with bounded
retries and verified TLS instead of recursing forever with ``verify=False``.
"""

from models.openai_compat_agent import OpenAICompatAgent


class ChatGPT(OpenAICompatAgent):
    def __init__(self, args, name):
        super().__init__(args, name)
