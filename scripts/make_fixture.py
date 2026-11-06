#!/usr/bin/env python3
"""Build tests/fixtures/qwen_outputs.jsonl from a real run JSON.

Usage:
    python3 scripts/make_fixture.py --latest
    python3 scripts/make_fixture.py ~/gsm8k_results/runs/baseline_2026-11-05_180000.json
    python3 scripts/make_fixture.py run.json -o tests/fixtures/qwen_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = Path.home() / "gsm8k_results" / "runs"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "qwen_outputs.jsonl"


def latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        sys.exit(f"error: no run files in {results_dir}")
    return runs[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="run JSON file")
    ap.add_argument("--latest", action="store_true", help="use the newest run")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--force", action="store_true",
                    help="write fixture even if the source run differs from 18/20")
    args = ap.parse_args()

    if args.latest:
        path = latest_run(Path(args.results_dir).expanduser())
    elif args.run:
        path = Path(args.run).expanduser()
    else:
        ap.error("specify a run file or --latest")

    with path.open("r", encoding="utf-8") as f:
        run = json.load(f)

    results = run.get("results", [])
    if len(results) < args.min_rows:
        sys.exit(
            f"error: run has {len(results)} rows, need >= {args.min_rows}\n"
            f"       rerun with: llm-quality run --label fixture --n {args.min_rows}"
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with output.open("w", encoding="utf-8") as f:
        for item in results:
            if item.get("infra_error"):
                sys.exit(
                    f"error: run has infra errors at idx={item['idx']}\n"
                    f"       cannot build a fixture from a contaminated run"
                )
            if item.get("pred") == item.get("gold"):
                correct += 1
            row = {"idx": item["idx"], "gold": item["gold"], "raw": item["raw"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(results)
    acc = correct / total * 100
    print(f"wrote {total} rows -> {output}")
    print(f"accuracy in source run: {correct}/{total} = {acc:.1f}%")

    if total != 20 or correct != 18:
        msg = (
            f"\n⚠ source run does not match the reference case:\n"
            f"    expected: 18/20 (90%)\n"
            f"    got:      {correct}/{total} ({acc:.1f}%)\n"
            f"\n"
            f"  Options:\n"
            f"    --force   write the fixture anyway (will break test_real_qwen_outputs_accuracy)\n"
            f"    (default) abort\n"
        )
        if not args.force:
            output.unlink(missing_ok=True)
            sys.exit(msg)
        print(msg + "  --force given, writing anyway", file=sys.stderr)