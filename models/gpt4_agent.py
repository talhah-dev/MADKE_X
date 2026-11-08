"""GPT-4 grading agent (OPTIONAL).

The MADKE-X pipeline computes EM, F1, Cons, Co2 and support rate automatically
and does not require this agent.  It remains available for the optional
``--gpt4_score`` evaluation path and uses the failure-aware client.
"""

from models.openai_compat_agent import OpenAICompatAgent


class GPT4(OpenAICompatAgent):
    def __init__(self, args, name):
        super().__init__(args, name)
