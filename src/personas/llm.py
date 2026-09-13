"""LangChain Runnable over llama-server's raw /completion endpoint (ADR 0007).

Deliberately NOT ChatOpenAI: /v1/chat/completions makes llama-server re-apply its own
Jinja template, which would hand prompt construction to the server and break the
train/inference parity guarantee in ADR 0006. Here we post our exact pre-rendered string.

In exchange for ~30 lines we keep .batch() with bounded concurrency, a retry policy, and
base-vs-tuned A/B as a single swapped base_url.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from langchain_core.runnables import Runnable, RunnableConfig

STOP = ["<|im_end|>", "<|endoftext|>"]


class LlamaCppCompletion(Runnable[str, str]):
    def __init__(self, base_url: str, decode: Any, timeout: float = 300.0,
                 max_retries: int = 3, client: Any | None = None):
        self.base_url = base_url.rstrip("/")
        self.decode = decode
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=timeout)

    def _body(self, prompt: str) -> dict:
        return {
            "prompt": prompt,
            "temperature": self.decode.temperature,
            "top_p": self.decode.top_p,
            "top_k": self.decode.top_k,
            "repeat_penalty": self.decode.repeat_penalty,
            "n_predict": self.decode.n_predict,
            "seed": self.decode.seed,
            "stop": STOP,
            # Prompts share a long system prefix; caching it across DIFFERENT prompts
            # would make results order-dependent, so it stays off for reproducibility.
            "cache_prompt": False,
        }

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs) -> str:  # noqa: A002
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(
                    f"{self.base_url}/completion",
                    json=self._body(input),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["content"]
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"llama-server at {self.base_url} failed after {self.max_retries} attempts"
        ) from last_error

    def close(self) -> None:
        self._client.close()


def wait_for_server(base_url: str, timeout: float = 300.0) -> None:
    """Block until llama-server reports ready, or raise."""
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    raise RuntimeError(f"llama-server at {base_url} did not become ready in {timeout}s")
