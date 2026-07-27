# -*- coding: utf-8 -*-
"""
PhysTrap unified LLM API client.

  - Follows a simple requests + Bearer-token calling convention compatible
    with most OpenAI-style chat-completion gateways.
  - The Solver role and Tool-agent roles can be routed to different models
    (they may also be the same model).
  - Automatic 429 rotation: on rate-limit, the client cycles through a pool
    of backup models; if a full cycle is exhausted it backs off for 1-3s
    before retrying, up to `max_retries` attempts in total.
  - When all models in the rotation are exhausted, an `ApiExhaustedError`
    (a soft/recoverable exception) is raised so that callers can mark a
    single sample as failed without aborting the overall run.

Configuration:
  Set the following environment variables before running any pipeline
  script (see README.md for details):
    PHYSTRAP_APP_KEY   - Bearer token / API key for your gateway
    PHYSTRAP_BASE_URL  - Chat-completion endpoint
                         (default: OpenAI-compatible "/chat/completions" path)
    PHYSTRAP_MODEL_POOL - comma-separated list of model names used for the
                          429 rotation pool (default: a 5-model example pool)
"""

import json
import os
import random
import re
import time

import requests


class ApiExhaustedError(RuntimeError):
    """Raised when every model in the rotation pool has exhausted its retries.

    Callers can catch this exception to mark the current sample as a soft
    failure without terminating the overall pipeline run.
    """


# Default endpoint; override with the PHYSTRAP_BASE_URL environment variable
# to point at your own OpenAI-compatible gateway.
DEFAULT_BASE_URL = "https://api.example.com/v1/chat/completions"

# Default rotation pool (5 models). Override with PHYSTRAP_MODEL_POOL
# (comma-separated) to use your own model names.
DEFAULT_MODEL_POOL = [
    "model-a",
    "model-b",
    "model-c",
    "model-d",
    "model-e",
]


def _load_model_pool():
    env_pool = os.getenv("PHYSTRAP_MODEL_POOL", "")
    if env_pool.strip():
        return [m.strip() for m in env_pool.split(",") if m.strip()]
    return list(DEFAULT_MODEL_POOL)


_MODEL_POOL = _load_model_pool()

# Models that only accept temperature=1 (forced override before sending).
# Populate via PHYSTRAP_TEMPERATURE_FIXED_1 (comma-separated) if needed.
_TEMPERATURE_FIXED_1 = set(
    m.strip() for m in os.getenv("PHYSTRAP_TEMPERATURE_FIXED_1", "").split(",") if m.strip()
)


class PhysTrapLLMClient:
    """
    Unified LLM API-calling client.

    Usage:
        client = PhysTrapLLMClient(app_key="your_key")
        msg = client.call_solver("question text ...")               # strong model answers
        data, msg = client.call_json_agent("tool", sys_prompt, user_prompt)  # lightweight JSON-output agent
    """

    def __init__(
        self,
        app_key: str = "",
        base_url: str = "",
        solver_model: str = "",
        tool_model: str = "",
        timeout: int = 600,
        max_retries: int = 30,
    ):
        self.app_key = app_key or os.getenv("PHYSTRAP_APP_KEY", "")
        self.base_url = (base_url or os.getenv("PHYSTRAP_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Each instance picks a random starting offset so that different
        # threads / seeds do not all hammer the same "first choice" model.
        offset = random.randint(0, len(_MODEL_POOL) - 1)
        self._model_rotation = _MODEL_POOL[offset:] + _MODEL_POOL[:offset]
        print(
            f"[PhysTrapLLMClient] rotation order for this instance: {' -> '.join(self._model_rotation)}",
            flush=True,
        )

        # Solver / tool default to position 0 of this instance's rotation,
        # but can be explicitly overridden by the caller.
        self.solver_model = solver_model or self._model_rotation[0]
        self.tool_model = tool_model or self._model_rotation[0]

    # ---- model selection ----

    def pick_model(self, role: str) -> str:
        """Return the model name for a role: 'solver' -> strong model, otherwise -> lightweight model."""
        return self.solver_model if role == "solver" else self.tool_model

    def _next_model(self, model: str) -> tuple:
        """
        Return (next_model_name, wrapped_around) following this instance's
        `_model_rotation` order. If `model` is not currently in the queue,
        start from index 0.
        """
        try:
            idx = self._model_rotation.index(model)
        except ValueError:
            idx = -1
        next_idx = (idx + 1) % len(self._model_rotation)
        wrapped = next_idx == 0  # completed a full cycle
        return self._model_rotation[next_idx], wrapped

    # ---- low-level request (with 429 rotation across backup models) ----

    def _request(self, model: str, messages: list, temperature: float, max_tokens: int) -> dict:
        """
        Send a chat-completion request.
        - 429: immediately switch to the next backup model in the queue (no wait).
          If a full cycle of all models still returns 429, sleep 1-3s before continuing.
        - 5xx / network errors: sleep a random 1-3s and retry (same model).
        - 4xx (other than 429): raise immediately, no retry.
        Retries up to `max_retries` times in total.
        The rotation order is decided randomly at instance construction time,
        so each instance (i.e. each parallel worker) uses a different order.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.app_key}",
        }

        current_model = model
        # Tracks consecutive 429 switches; only sleeps after a full cycle.
        consecutive_429 = 0

        for attempt in range(1, self.max_retries + 1):
            # Some models only accept temperature=1; auto-adapt.
            effective_temperature = 1 if current_model in _TEMPERATURE_FIXED_1 else temperature
            payload = {
                "model": current_model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": effective_temperature,
            }
            ts = time.strftime("%H:%M:%S")
            print(
                f"  [{ts}] -> sending request model={current_model}"
                f" attempt={attempt}/{self.max_retries}"
                f" temperature={effective_temperature} max_tokens={max_tokens}",
                flush=True,
            )
            try:
                t0 = time.time()
                resp = requests.post(
                    self.base_url, json=payload, headers=headers,
                    timeout=(10, self.timeout),
                )
                elapsed = round(time.time() - t0, 1)
                if resp.status_code == 200:
                    print(f"  [{ts}] OK model={current_model} elapsed={elapsed}s", flush=True)
                    result = resp.json()
                    result["_used_model"] = current_model  # record the model actually used
                    return result
                elif resp.status_code == 429:
                    next_m, wrapped = self._next_model(current_model)
                    consecutive_429 += 1
                    if wrapped:
                        # Completed a full cycle; all models are rate-limited, back off and retry.
                        wait = random.uniform(1.0, 3.0)
                        print(
                            f"  [{ts}] [429 x full-cycle/{consecutive_429}] all {len(_MODEL_POOL)} models rate-limited"
                            f" attempt={attempt}/{self.max_retries}"
                            f" -> waiting {wait:.1f}s, next={next_m}",
                            flush=True,
                        )
                        time.sleep(wait)
                        consecutive_429 = 0
                    else:
                        # There are still untried models; switch immediately.
                        print(
                            f"  [{ts}] [429] model={current_model} attempt={attempt}/{self.max_retries}"
                            f" -> switching to {next_m}",
                            flush=True,
                        )
                    current_model = next_m
                elif 400 <= resp.status_code < 500:
                    # Non-429 4xx client errors (e.g. 401 auth failure, 400 bad request): fail without retry.
                    print(
                        f"  [{ts}] [HTTP {resp.status_code}] model={current_model}"
                        f" client error (no retry): {resp.text[:200]}",
                        flush=True,
                    )
                    raise RuntimeError(f"API client error HTTP {resp.status_code} (no retry): {resp.text[:300]}")
                else:
                    # 5xx server error, retryable.
                    consecutive_429 = 0
                    wait = random.uniform(1.0, 3.0)
                    print(
                        f"  [{ts}] [HTTP {resp.status_code}] model={current_model}"
                        f" server error -> waiting {wait:.1f}s and retrying: {resp.text[:200]}",
                        flush=True,
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
            except RuntimeError:
                raise
            except Exception as e:
                consecutive_429 = 0
                wait = random.uniform(1.0, 3.0)
                print(
                    f"  [{ts}] [request exception] model={current_model} attempt={attempt}/{self.max_retries}"
                    f" error={e} -> waiting {wait:.1f}s and retrying",
                    flush=True,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)

        raise ApiExhaustedError(
            f"API call failed: model={model}, retried {self.max_retries} times, "
            "all models rate-limited or erroring"
        )

    # ---- response parsing ----

    @staticmethod
    def _extract_message(response: dict) -> dict:
        """
        Extract the message field from an API response, tolerating several
        response formats:
        - reasoning models: separate `content` + `reasoning_content` fields
        - some models: `content` empty, everything is in `reasoning_content`
        - some models: `reasoning_content` empty, `<think>...</think>` is
          embedded directly inside `content`
        """
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            msg = choice.get("message", {})
            reasoning = msg.get("reasoning_content") or ""
            content = msg.get("content") or ""
            finish_reason = choice.get("finish_reason", "")

            # Some models embed <think>...</think> inside content; strip it out.
            if not reasoning and "<think>" in content:
                import re as _re
                think_match = _re.search(r"<think>(.*?)</think>", content, _re.S)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    content = _re.sub(r"<think>.*?</think>\s*", "", content, flags=_re.S).strip()

            full = ""
            if reasoning:
                full += f"<think_reasoning>\n{reasoning}\n</think_reasoning>\n"
            full += content
            if finish_reason == "length":
                full += "\n\n[System note: output truncated at max_tokens limit.]\n"
            return {
                "reasoning": reasoning,
                "content": content,
                "full": full,
                "finish_reason": finish_reason,
            }
        elif "content" in response:
            return {"reasoning": "", "content": response["content"],
                    "full": response["content"], "finish_reason": ""}
        return {"reasoning": "", "content": "", "full": "", "finish_reason": ""}

    @staticmethod
    def _parse_json(text: str):
        """Best-effort extraction of a JSON object from free text."""
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        return {}

    # ---- public interface ----

    def call_solver(self, prompt: str) -> dict:
        """
        Solver: have the strong model attempt to answer a question (user message only).
        The returned dict contains a 'model' field recording the model that
        actually produced the answer (may differ from the initial choice due
        to 429 rotation).
        """
        model = self.pick_model("solver")
        start = time.time()
        raw = self._request(model, [{"role": "user", "content": prompt}],
                             temperature=0.3, max_tokens=8192)
        # `_used_model` was injected at the end of `_request`.
        actual_model = raw.pop("_used_model", model)
        msg = self._extract_message(raw)
        msg["time"] = round(time.time() - start, 2)
        msg["model"] = actual_model  # record the model actually used
        return msg

    def call_json_agent(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = None,
    ) -> tuple:
        """
        JSON agent: system + user two-message call, expecting JSON output.
        Returns (parsed_dict, raw_message_dict).
        """
        model = self.pick_model(role)
        strict_suffix = (
            "\n\nPlease output only a single JSON object. Do not output Markdown "
            "or any explanatory text. If you are unsure of a field name, still "
            "use the English keys specified by the task."
        )
        messages = [
            {"role": "system", "content": system_prompt + strict_suffix},
            {"role": "user", "content": user_prompt},
        ]
        start = time.time()
        raw = self._request(model, messages, temperature=temperature, max_tokens=max_tokens)
        actual_model = raw.pop("_used_model", model)
        msg = self._extract_message(raw)
        msg["time"] = round(time.time() - start, 2)
        msg["model"] = actual_model  # record the model actually used

        text = msg.get("content", "") or msg.get("full", "")
        data = self._parse_json(text)
        if not isinstance(data, dict):
            data = {"raw_content": text, "parse_error": True}
        return data, msg
