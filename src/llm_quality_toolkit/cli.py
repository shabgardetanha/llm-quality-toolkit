"""Command-line interface for llm-quality-toolkit."""

from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import sys
from pathlib import Path

import requests

from ._util import redact_sensitive
from .comparator import load_runs, to_markdown
from .runner import run, save, summarize

DEFAULT_ENDPOINT = (
    os.environ.get("LLM_ENDPOINT")
    or os.environ.get("LLAMA_ENDPOINT")
    or "http://127.0.0.1:8080/v1/chat/completions"
)
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or "Qwen3.5-4B"
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("LLAMA_API_KEY")
DEFAULT_DATA = Path.home() / "gsm8k_test.jsonl"
DEFAULT_RESULTS_DIR = Path.home() / "gsm8k_results" / "runs"


def _parse_headers(items: list[str]) -> dict:
    out = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"invalid header {item!r}: expected 'Key: Value'")
        k, v = item.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _validate_run_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise SystemExit(f"--n must be positive, got {args.n}")
    if args.max_tokens <= 0:
        raise SystemExit(f"--max-tokens must be positive, got {args.max_tokens}")
    if not (0.0 <= args.temperature <= 2.0):
        raise SystemExit(f"--temperature must be in [0, 2], got {args.temperature}")
    if not args.label or not args.label.strip():
        raise SystemExit("--label cannot be empty")
    if any(c in args.label for c in "/\\"):
        raise SystemExit(f"--label cannot contain path separators: {args.label!r}")
    if any(ord(c) < 32 for c in args.label):
        raise SystemExit("--label cannot contain control characters")
    if args.max_retries < 1:
        raise SystemExit(f"--max-retries must be >= 1, got {args.max_retries}")


def _cmd_run(args: argparse.Namespace) -> None:
    try:
        _validate_run_args(args)
        extra_headers = _parse_headers(args.header) if args.header else None
        r = run(
            label=args.label,
            endpoint=args.endpoint,
            model=args.model,
            data_path=Path(args.data).expanduser(),
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            notes=args.notes,
            server_args=args.server_args,
            results_dir=Path(args.results_dir).expanduser(),
            api_key=args.api_key,
            extra_headers=extra_headers,
            max_retries=args.max_retries,
        )
        out = save(r, Path(args.results_dir).expanduser())
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except requests.RequestException as e:
        print(f"network error: {redact_sensitive(str(e))}", file=sys.stderr)
        sys.exit(3)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"unexpected error: {redact_sensitive(f'{type(e).__name__}: {e}')}",
              file=sys.stderr)
        sys.exit(1)

    s = r["summary"]
    print(
        f"\n{s['accuracy_pct']:.1f}% ({s['correct']}/{s['valid']} valid)",
        file=sys.stderr,
    )
    if s["infra_errors"]:
        print(f"warning: {s['infra_errors']} infra error(s)", file=sys.stderr)
    if s["extraction_failures"]:
        print(f"warning: {s['extraction_failures']} extraction failure(s)",
              file=sys.stderr)
    if s["gold_extraction_failures"]:
        print(f"warning: {s['gold_extraction_failures']} gold extraction failure(s)",
              file=sys.stderr)
    print(f"saved: {out}", file=sys.stderr)


def _cmd_compare(args: argparse.Namespace) -> None:
    labels = args.labels.split(",") if args.labels else None
    runs = load_runs(Path(args.results_dir).expanduser(), labels)
    if not runs:
        print("no runs found", file=sys.stderr)
        sys.exit(1)
    md = to_markdown(runs)
    if args.out:
        Path(args.out).expanduser().write_text(md, encoding="utf-8")
        print(f"saved: {args.out}", file=sys.stderr)
    else:
        print(md)


def _cmd_list(args: argparse.Namespace) -> None:
    runs = load_runs(Path(args.results_dir).expanduser())
    if not runs:
        print("no runs found", file=sys.stderr)
        return
    print(f"{len(runs)} runs in {args.results_dir}:")
    for r in runs:
        s = r["summary"]
        line = (
            f"  {r['label']:25s} {s['accuracy_pct']:6.1f}%  "
            f"({s['correct']}/{s.get('valid', s['total'])})  {r['timestamp'][:16]}"
        )
        if s.get("infra_errors"):
            line += f"  [!{s['infra_errors']} infra]"
        print(line)


def _cmd_re_extract(args: argparse.Namespace) -> None:
    from .extractor import extract_answer

    files = sorted(set(sum((_glob.glob(p) for p in args.patterns), [])))
    if not files:
        print("no files matched", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(args.results_dir).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"skip {path}: {e}", file=sys.stderr)
            continue
        for item in r["results"]:
            if item.get("infra_error"):
                item["pred"] = None
                item["correct"] = False
                continue
            item["pred"] = extract_answer(item.get("raw", ""))
            if item.get("gold_raw"):
                item["gold"] = extract_answer(item["gold_raw"])
                item["gold_extracted"] = item["gold"] is not None
            item["correct"] = (
                item["pred"] is not None and item["pred"] == item["gold"]
            )
        r["summary"] = summarize(r["results"], r["summary"].get("total_time_s", 0.0))
        r["label"] = args.label or (r["label"] + "__reextract")
        out = save(r, results_dir)
        print(
            f"  {Path(path).name} -> {r['summary']['accuracy_pct']:.1f}%",
            file=sys.stderr,
        )
        print(f"  saved: {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(prog="llm-quality")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run GSM8K against a live server")
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--header", action="append", default=[], metavar="K:V")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--notes", default="")
    p.add_argument("--server-args", default="")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("compare")
    p.add_argument("--labels")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("list")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("re-extract")
    p.add_argument("patterns", nargs="+")
    p.add_argument("--label")
    p.set_defaults(func=_cmd_re_extract)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()