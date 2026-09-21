from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from .cases import build_suite
from .metrics import analyze
from .report import generate_report
from .runner import run_suite
from .util import load_config


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply explicit CLI overrides without mutating configuration files on disk."""
    run = cfg.setdefault("run", {})
    api = cfg.setdefault("api", {})

    if getattr(args, "profile", None):
        run["profile"] = args.profile
    if getattr(args, "output_dir", None):
        run["output_dir"] = args.output_dir
    if getattr(args, "concurrency", None) is not None:
        api["concurrency"] = int(args.concurrency)
    if getattr(args, "model", None):
        api["model"] = args.model
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(prog="jevbench")
    p.add_argument("--config", default="config/architecture.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate/inspect a benchmark suite without API calls")
    g.add_argument("--list", action="store_true", help="List experiment counts")
    g.add_argument("--profile", help="Override run.profile from the YAML config")

    r = sub.add_parser("run", help="Run a benchmark against TypeSafe")
    r.add_argument("--only", action="append", default=[], help="Run only this experiment; may be repeated")
    r.add_argument("--profile", help="Override run.profile from the YAML config")
    r.add_argument("--concurrency", type=int, help="Override api.concurrency from the YAML config")
    r.add_argument("--model", help="Override api.model from the YAML config")
    r.add_argument("--output-dir", help="Override run.output_dir from the YAML config")

    a = sub.add_parser("analyze", help="Analyze an existing run directory")
    a.add_argument("run_dir")

    rep = sub.add_parser("report", help="Generate Markdown report/figures for an analyzed run")
    rep.add_argument("run_dir")

    args = p.parse_args()
    cfg = _apply_overrides(load_config(args.config), args)

    if args.cmd == "generate":
        cases = build_suite(cfg)
        print(f"profile={cfg['run']['profile']} cases={len(cases)}")
        if args.list:
            counts: dict[str, int] = {}
            for c in cases:
                counts[c.experiment] = counts.get(c.experiment, 0) + 1
            for k in sorted(counts):
                print(f"{k:36s} {counts[k]:5d}")
    elif args.cmd == "run":
        only = set(args.only) if args.only else None
        run_dir = asyncio.run(run_suite(cfg, only))
        print(f"RUN_DIR={run_dir}")
        analyze(run_dir)
        report = generate_report(run_dir)
        print(f"REPORT={report}")
    elif args.cmd == "analyze":
        analyze(args.run_dir)
        print(Path(args.run_dir) / "summary.json")
    elif args.cmd == "report":
        print(generate_report(args.run_dir))


if __name__ == "__main__":
    main()
