from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .api import TypeSafeAPI
from .cases import build_suite
from .util import append_jsonl, json_dump


async def run_suite(cfg: dict[str, Any], selected_experiments: set[str] | None = None) -> Path:
    api = TypeSafeAPI(cfg["api"])
    cases = build_suite(cfg)
    if selected_experiments:
        cases = [c for c in cases if c.experiment in selected_experiments]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg["run"]["output_dir"]) / f"run_{ts}_{cfg['run']['profile']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    cases_path = run_dir / "cases.jsonl"

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": api.model,
        "profile": cfg["run"]["profile"],
        "seed": cfg["run"]["seed"],
        "case_count": len(cases),
        "python": sys.version,
        "platform": platform.platform(),
        "config": cfg,
    }
    json_dump(manifest_path, manifest)
    for c in cases:
        append_jsonl(cases_path, c.to_dict())

    sem = asyncio.Semaphore(int(cfg["api"].get("concurrency", 4)))
    completed = 0
    lock = asyncio.Lock()
    started = time.perf_counter()

    async with httpx.AsyncClient(http2=True) as client:
        async def one(case):
            nonlocal completed
            async with sem:
                result = await api.call(client, case.state, case.questions)
            row = {
                "case_id": case.case_id,
                "experiment": case.experiment,
                "metadata": case.metadata,
                "expected": case.expected,
                "request": {
                    "state": case.state,
                    "questions": case.questions,
                    "model": api.model,
                } if cfg["run"].get("save_request_bodies", True) else None,
                "ok": result.ok,
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "attempts": result.attempts,
                "response": result.response_json if cfg["run"].get("save_response_bodies", True) else None,
                "response_headers": result.response_headers,
                "error": result.error,
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            async with lock:
                append_jsonl(raw_path, row)
                completed += 1
                if completed % 10 == 0 or completed == len(cases):
                    elapsed = time.perf_counter() - started
                    print(f"[{completed}/{len(cases)}] elapsed={elapsed:.1f}s last={case.experiment} ok={result.ok} {result.latency_ms:.1f}ms")

        await asyncio.gather(*(one(c) for c in cases))

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = time.perf_counter() - started
    json_dump(manifest_path, manifest)
    return run_dir
