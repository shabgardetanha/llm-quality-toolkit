"""Run GSM8K evaluation against a live server."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from ._util import redact_sensitive
from .extractor import extract_answer
from .props import fetch_props


RETRY_BACKOFF_S = (1.0, 2.0, 4.0)


def _load_questions(path: Path, n: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"GSM8K data not found: {path}")
    data: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if len(data) >= n:
                break
    if not data:
        raise ValueError(f"GSM8K file is empty: {path}")
    return data


def _build_headers(api_key: Optional[str], extra_headers: Optional[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _ask(
    endpoint: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: Optional[str],
    extra_headers: Optional[dict],
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(
        endpoint,
        json=payload,
        headers=_build_headers(api_key, extra_headers),
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ask_with_retry(
    endpoint: str,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: Optional[str],
    extra_headers: Optional[dict],
    max_retries: int,
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(raw, error)``. Exactly one is None on return."""
    if max_retries < 1:
        max_retries = 1
    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            return _ask(
                endpoint, model, question, temperature, max_tokens, timeout,
                api_key, extra_headers,
            ), None
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if 400 <= status < 500:
                return None, f"HTTP {status}"
            last_error = f"HTTP {status}"
        except requests.Timeout:
            last_error = "timeout"
        except requests.ConnectionError:
            last_error = "connection error"
        except Exception as e:
            return None, redact_sensitive(f"{type(e).__name__}: {e}")
        if attempt < max_retries - 1:
            time.sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
    return None, last_error or "unknown error"


def summarize(results: list[dict], total_time_s: float) -> dict:
    """Compute summary counters, distinguishing infra errors from model errors."""
    total = len(results)
    infra = sum(1 for r in results if r.get("infra_error"))
    valid = total - infra
    correct = sum(1 for r in results if r.get("correct"))
    extraction_failures = sum(
        1 for r in results
        if not r.get("infra_error") and r.get("pred") is None
    )
    gold_failures = sum(1 for r in results if not r.get("gold_extracted", True))

    warnings: list[str] = []
    if infra:
        warnings.append(
            f"{infra} infra error(s) — accuracy computed over {valid} valid run(s)"
        )
    if extraction_failures:
        warnings.append(
            f"{extraction_failures} extraction failure(s) on non-error responses"
        )
    if gold_failures:
        warnings.append(
            f"{gold_failures} gold answer(s) could not be extracted — accuracy meaningless"
        )

    return {
        "correct": correct,
        "valid": valid,
        "total": total,
        "accuracy_pct": round(correct / valid * 100, 2) if valid else 0.0,
        "accuracy_including_errors_pct": round(correct / total * 100, 2) if total else 0.0,
        "infra_errors": infra,
        "extraction_failures": extraction_failures,
        "gold_extraction_failures": gold_failures,
        "total_time_s": round(total_time_s, 1),
        "avg_time_per_q_s": round(total_time_s / total, 2) if total else 0.0,
        "warnings": warnings,
    }


def run(
    *,
    label: str,
    endpoint: str,
    model: str,
    data_path: Path,
    n: int,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 300,
    notes: str = "",
    server_args: str = "",
    results_dir: Path,
    api_key: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    max_retries: int = 3,
    progress: bool = True,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(data_path, n)

    props_result = fetch_props(endpoint, api_key=api_key, extra_headers=extra_headers)
    server_props = props_result["data"] if props_result["status"] == "ok" else None
    if progress:
        if server_props:
            print("props: OK", file=sys.stderr)
        else:
            print(f"props: {props_result['error']}", file=sys.stderr)

    iterator = questions
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(questions, desc=f"GSM8K [{label}]", unit="q", file=sys.stderr)
        except ImportError:
            pass

    results: list[dict] = []
    t_start = time.time()

    for i, ex in enumerate(iterator):
        q = ex["question"]
        gold_raw = ex["answer"]
        gold = extract_answer(gold_raw)

        t0 = time.time()
        raw, error = _ask_with_retry(
            endpoint, model, q, temperature, max_tokens, timeout,
            api_key, extra_headers, max_retries,
        )
        elapsed = time.time() - t0

        if error:
            pred = None
            infra_error = True
            correct = False
        else:
            pred = extract_answer(raw)
            infra_error = False
            correct = pred is not None and pred == gold

        results.append(
            {
                "idx": i,
                "question": q,
                "gold": gold,
                "gold_raw": gold_raw,
                "gold_extracted": gold is not None,
                "pred": pred,
                "correct": correct,
                "infra_error": infra_error,
                "infra_error_message": error,
                "elapsed_s": round(elapsed, 2),
                "raw": raw if raw is not None else f"<INFRA_ERROR: {error}>",
            }
        )

    total_time = time.time() - t_start

    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": endpoint,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
        "notes": notes,
        "server_args_manual": server_args,
        "server_props": server_props,
        "server_props_status": props_result["status"],
        "server_props_error": props_result["error"],
        "api_auth": bool(api_key),
        "extra_headers_used": sorted((extra_headers or {}).keys()),
        "extractor_version": "v3",
        "results": results,
        "summary": summarize(results, total_time),
    }


def save(run_result: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = results_dir / f"{run_result['label']}_{ts}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(run_result, f, ensure_ascii=False, indent=2)
    return out