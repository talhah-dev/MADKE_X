"""Failure-aware client for any OpenAI-compatible chat-completions endpoint (fix C7/C8).

Works with vLLM, Ollama (``/v1``), llama.cpp server, LM Studio, OpenAI, Azure
OpenAI (``--endpoint`` + ``--deployment_id``) and similar servers.  Only the
Python standard library is used (``urllib`` + ``ssl``), so no extra dependency
is required on Kaggle / Colab / a local machine.

Behaviour
---------
* bounded retries (``max_retries``) with exponential back-off on network errors,
  time-outs, HTTP 408/409/425/429 and 5xx responses;
* non-retryable 4xx responses fail immediately;
* TLS is verified by default; ``ca_bundle`` points at a custom CA file and
  ``insecure=True`` disables verification (loudly, for local self-signed servers);
* every call returns :class:`models.base.AgentResult` -- ``ok=False`` carries
  the error message and the number of attempts instead of an error string that
  could be mistaken for an answer.
"""

import json
import ssl
import time
import urllib.error
import urllib.request
import warnings

from models.base import AgentResult, BaseAgent

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _getattr(args, name, default=None):
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


class OpenAICompatAgent(BaseAgent):
    def __init__(self, args=None, name="Agent", **overrides):
        super().__init__(name)
        cfg = {
            "base_url": _getattr(args, "base_url", None),
            "api_key": _getattr(args, "api_key", None) or "EMPTY",
            "model": _getattr(args, "model", None),
            "endpoint": _getattr(args, "endpoint", None),
            "deployment_id": _getattr(args, "deployment_id", None),
            "api_version": _getattr(args, "api_version", "2023-07-01-preview"),
            "temperature": _getattr(args, "temperature", 0.5),
            "top_p": _getattr(args, "top_p", None),
            "max_tokens": _getattr(args, "max_tokens", None),
            "timeout": _getattr(args, "timeout", 120),
            "max_retries": _getattr(args, "max_retries", 3),
            "backoff": _getattr(args, "backoff", 2.0),
            "ca_bundle": _getattr(args, "ca_bundle", None),
            "insecure": bool(_getattr(args, "insecure", False)),
            "extra_headers": _getattr(args, "extra_headers", None) or {},
            "verbose": bool(_getattr(args, "verbose", False)),
        }
        cfg.update({k: v for k, v in overrides.items() if v is not None})

        self.temperature = cfg["temperature"]
        self.top_p = cfg["top_p"]
        self.max_tokens = cfg["max_tokens"]
        self.timeout = cfg["timeout"]
        self.max_retries = int(cfg["max_retries"])
        self.backoff = float(cfg["backoff"])
        self.verbose = cfg["verbose"]
        self.model = cfg["model"]

        if cfg["endpoint"] and cfg["deployment_id"]:
            # Azure OpenAI deployment (original MADKE set-up)
            self.url = ("https://{endpoint}.openai.azure.com/openai/deployments/{deployment_id}"
                        "/chat/completions?api-version={api_version}").format(
                endpoint=cfg["endpoint"], deployment_id=cfg["deployment_id"], api_version=cfg["api_version"])
            self.headers = {"Content-Type": "application/json", "api-key": cfg["api_key"]}
        else:
            base_url = (cfg["base_url"] or "http://localhost:8000/v1").rstrip("/")
            if not base_url.endswith("/chat/completions"):
                base_url = base_url + "/chat/completions"
            self.url = base_url
            self.headers = {"Content-Type": "application/json", "Authorization": "Bearer {}".format(cfg["api_key"])}
        self.headers.update(cfg["extra_headers"])

        self.ssl_context = None
        if self.url.lower().startswith("https://"):
            if cfg["insecure"]:
                warnings.warn("TLS certificate verification DISABLED (--insecure) for {}".format(self.url))
                self.ssl_context = ssl.create_default_context()
                self.ssl_context.check_hostname = False
                self.ssl_context.verify_mode = ssl.CERT_NONE
            else:
                self.ssl_context = ssl.create_default_context(cafile=cfg["ca_bundle"])

    # ------------------------------------------------------------------
    def _payload(self, messages):
        payload = {"messages": messages, "temperature": self.temperature}
        if self.model:
            payload["model"] = self.model
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _post_once(self, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
            return response.status, response.read().decode("utf-8", errors="replace")

    @staticmethod
    def _extract(body):
        """Return ``(text, usage)`` or raise ``ValueError`` on an unusable completion."""
        completion = json.loads(body)
        if "error" in completion and "choices" not in completion:
            err = completion["error"]
            raise ValueError("API error: {}".format(err.get("message", err) if isinstance(err, dict) else err))
        choices = completion.get("choices") or []
        if not choices:
            raise ValueError("completion has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None and "text" in choices[0]:
            content = choices[0]["text"]
        if not isinstance(content, str):
            raise ValueError("completion content is not a string")
        return content, completion.get("usage") or {}

    def generate(self, messages):
        payload = self._payload(messages)
        last_error = None
        start = time.time()
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            try:
                status, body = self._post_once(payload)
                text, usage = self._extract(body)
                return AgentResult(ok=True, text=text, attempts=attempts, latency=time.time() - start, usage=usage)
            except urllib.error.HTTPError as ex:
                body = ""
                try:
                    body = ex.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                last_error = "HTTP {} {}: {}".format(ex.code, ex.reason, body)
                if ex.code not in RETRYABLE_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as ex:
                last_error = "network error: {}".format(ex)
            except (ValueError, json.JSONDecodeError) as ex:
                last_error = "bad response: {}".format(ex)
            if attempt < self.max_retries:
                delay = self.backoff ** attempt
                if self.verbose:
                    print("[{}] attempt {} failed ({}); retrying in {:.1f}s".format(self.agent_name, attempts, last_error, delay))
                time.sleep(delay)
        return AgentResult(ok=False, text="", error=last_error or "unknown error", attempts=attempts,
                           latency=time.time() - start)


def preflight(args, retries=1):
    """Send one tiny chat request before a live run.

    Returns ``(ok, message)``.  A live run must not start when this fails: without
    a reachable server every call fails and no model output is produced.
    """
    agent = OpenAICompatAgent(args, "Preflight", max_retries=retries, temperature=0.0, max_tokens=16)
    res = agent([{"role": "user", "content": "Reply with the single word: OK"}])
    if res.ok and res.text.strip():
        return True, "endpoint {} answered with model {!r}: {!r}".format(agent.url, agent.model, res.text.strip()[:60])
    if res.ok:
        return False, "endpoint {} returned an EMPTY reply for model {!r}".format(agent.url, agent.model)
    return False, ("endpoint {} is not usable for model {!r}: {}\n"
                   "  -> Is the model server running (vLLM / Ollama) and is --base_url / --model correct?").format(
        agent.url, agent.model, res.error)


def build_agent(args, name, mode=None):
    """Factory: ``ScriptedAgent`` for ``--mode offline`` else :class:`OpenAICompatAgent`."""
    mode = mode or _getattr(args, "mode", "live")
    if mode == "offline":
        from models.scripted_agent import ScriptedAgent
        return ScriptedAgent(name, seed=_getattr(args, "seed", 0), fail_rate=_getattr(args, "fail_rate", 0.0),
                             dataset_name=_getattr(args, "dataset_name", "triviaqa"))
    return OpenAICompatAgent(args, name)
