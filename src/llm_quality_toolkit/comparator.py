"""Compare multiple runs and render Markdown."""

from __future__ import annotations

import glob as _glob
import json
from datetime import datetime
from pathlib import Path


def load_runs(results_dir: Path, labels: list[str] | None = None) -> list[dict]:
    """Load all run JSON files from ``results_dir``, optionally filtered."""
    runs: list[dict] = []
    for f in sorted(_glob.glob(str(results_dir / "*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            continue
        if labels and r.get("label") not in labels:
            continue
        r["_file"] = Path(f).name
        runs.append(r)
    return runs


def _fmt_config(r: dict) -> str:
    props = r.get("server_props") or {}
    gen = props.get("default_generation_settings") or {}
    lines = [f"- model: `{props.get('model_alias') or r.get('model', '?')}`"]
    if props.get("build_info"):
        lines.append(f"- build: `{props['build_info']}`")
    if gen:
        parts = [
            f"{k}={gen[k]}"
            for k in ("n_ctx", "temperature", "top_k", "top_p", "min_p")
            if k in gen
        ]
        if parts:
            lines.append(f"- generation: `{' '.join(parts)}`")
    if r.get("notes"):
        lines.append(f"- notes: {r['notes']}")
    if r.get("server_args_manual"):
        lines.append(f"- server_args (manual): `{r['server_args_manual']}`")
    if not props and not r.get("server_args_manual"):
        lines.append("- warning: no server config was recorded")
    return "\n".join(lines)


def to_markdown(runs: list[dict]) -> str:
    """Render a comparison report in Markdown."""
    out = [
        "# GSM8K comparison report\n",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "## Summary\n",
        "| Label | Timestamp | N | Accuracy | Correct | s/q | Notes |",
        "|-------|-----------|---|----------|---------|-----|-------|",
    ]
    for r in runs:
        s = r["summary"]
        valid = s.get("valid", s["total"])
        out.append(
            f"| `{r['label']}` | {r['timestamp'][:16]} | {s['total']} "
            f"| **{s['accuracy_pct']:.1f}%** | {s['correct']}/{valid} "
            f"| {s['avg_time_per_q_s']:.1f} | {r.get('notes', '')} |"
        )

    # Data-quality warnings
    warning_lines: list[str] = []
    for r in runs:
        s = r["summary"]
        ie = s.get("infra_errors", 0)
        ef = s.get("extraction_failures", 0)
        gf = s.get("gold_extraction_failures", 0)
        parts: list[str] = []
        if ie:
            parts.append(f"{ie} infra error(s)")
        if ef:
            parts.append(f"{ef} extraction failure(s)")
        if gf:
            parts.append(f"{gf} gold extraction failure(s)")
        if parts:
            warning_lines.append(f"- `{r['label']}`: " + ", ".join(parts))

    if warning_lines:
        out += ["", "## ⚠️ Data quality warnings\n"] + warning_lines + [""]

    if len(runs) >= 2:
        n = min(r["summary"]["total"] for r in runs)
        out += [
            "",
            f"## Per-question disagreements (first {n})\n",
            "| idx | gold | " + " | ".join(r["label"] for r in runs) + " |",
            "|-----|------|" + "|".join("---" for _ in runs) + "|",
        ]
        diff = 0
        for i in range(n):
            corrects = [
                r["results"][i]["correct"] if i < len(r["results"]) else None
                for r in runs
            ]
            if len(set(corrects)) <= 1:
                continue
            diff += 1
            gold = runs[0]["results"][i]["gold"]
            cells = []
            for r in runs:
                if i >= len(r["results"]):
                    cells.append("—")
                    continue
                res = r["results"][i]
                if res.get("infra_error"):
                    mark = "⚠"
                elif res["correct"]:
                    mark = "✓"
                else:
                    mark = "✗"
                cells.append(f"{mark} `{res['pred']}`")
            out.append(f"| {i} | {gold} | " + " | ".join(cells) + " |")
        if diff == 0:
            out.append("| — | — | " + " | ".join("all agree" for _ in runs) + " |")

    out += ["", "## Configuration per run\n"]
    for r in runs:
        out.append(f"### `{r['label']}`")
        out.append(f"- timestamp: {r['timestamp']}")
        out.append(f"- file: `{r['_file']}`")
        out.append(f"- extractor: {r.get('extractor_version', '?')}")
        out.append(_fmt_config(r))
        out.append("")
    return "\n".join(out)