from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResult:
    ok: bool
    status_code: int | None
    latency_ms: float
    response_json: dict[str, Any] | None
    response_text: str | None
    response_headers: dict[str, str]
    error: str | None
    attempts: int


class TypeSafeAPI:
    def __init__(self, cfg: dict[str, Any]):
        self.base_url = cfg["base_url"].rstrip("/")
        self.endpoint = cfg["endpoint"]
        self.model = cfg["model"]
        self.timeout = float(cfg.get("timeout_seconds", 60))
        self.max_retries = int(cfg.get("max_retries", 5))
        self.min_retry = float(cfg.get("min_retry_seconds", 1.0))
        self.max_retry = float(cfg.get("max_retry_seconds", 30.0))
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is not set")

    async def call(self, client: httpx.AsyncClient, state: Any, questions: dict[str, Any]) -> ApiResult:
        payload = {"state": state, "model": self.model, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "jev-blackbox-benchmark/0.1",
        }
        last_error = None
        start_all = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = await client.post(
                    self.base_url + self.endpoint,
                    headers=headers,
                    content=json.dumps(payload, ensure_ascii=False),
                    timeout=self.timeout,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                h = {k.lower(): v for k, v in resp.headers.items()}
                try:
                    body = resp.json()
                except Exception:
                    body = None
                if 200 <= resp.status_code < 300:
                    return ApiResult(True, resp.status_code, latency_ms, body, resp.text, h, None, attempt)
                last_error = f"HTTP {resp.status_code}: {resp.text[:1000]}"
                if resp.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    return ApiResult(False, resp.status_code, latency_ms, body, resp.text, h, last_error, attempt)
                retry_after = h.get("retry-after")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = self.min_retry
                else:
                    delay = min(self.max_retry, self.min_retry * (2 ** (attempt - 1)))
                    delay *= random.uniform(0.8, 1.2)
                await asyncio.sleep(delay)
            except Exception as exc:
                last_error = repr(exc)
                if attempt < self.max_retries:
                    delay = min(self.max_retry, self.min_retry * (2 ** (attempt - 1)))
                    await asyncio.sleep(delay * random.uniform(0.8, 1.2))
        return ApiResult(
            False,
            None,
            (time.perf_counter() - start_all) * 1000,
            None,
            None,
            {},
            last_error,
            self.max_retries,
        )
