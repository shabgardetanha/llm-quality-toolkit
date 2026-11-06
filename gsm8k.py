#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ابزار یکپارچه GSM8K برای llama.cpp / سرورهای سازگار با OpenAI.

دستورات:
    run         اجرای تست و ذخیرهٔ ساختاریافته
    compare     مقایسهٔ چند اجرا و تولید Markdown
    list        فهرست اجراهای ذخیره‌شده
    re-extract  بازپردازش raw با استخراج جدید (بدون فراخوانی مدل)

نمونه‌ها:
    python3 gsm8k.py run --label baseline --n 20
    python3 gsm8k.py run --label q4kv --n 20 --notes "cache-type-k=q4_0"
    python3 gsm8k.py compare
    python3 gsm8k.py compare --labels baseline,q4kv --out report.md
    python3 gsm8k.py list
    python3 gsm8k.py re-extract '~/gsm8k_results/runs/baseline_*.json'
"""
import argparse, glob as _glob, json, os, re, signal, sys, time, urllib.request
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr); sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: pip install tqdm", file=sys.stderr); sys.exit(1)


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL    = "Qwen3.5-4B"
DEFAULT_DATA     = os.path.expanduser("~/gsm8k_test.jsonl")
RESULTS_DIR      = os.path.expanduser("~/gsm8k_results/runs")


# ============ استخراج پاسخ ============
def _clean(s):
    return s.replace(",", "").rstrip(".").strip()


def extract_answer(text):
    """ترتیب: #### N → \\boxed{} (آخرین) → **bold** (آخرین) → آخرین عدد."""
    if not text:
        return None
    m = re.search(r'####\s*(-?\d[\d,\.]*)', text)
    if m:
        return _clean(m.group(1))
    for b in reversed(re.findall(r'\\boxed\s*\{([^}]*)\}', text)):
        n = re.findall(r'-?\d[\d,\.]*', b)
        if n:
            return _clean(n[-1])
    for b in reversed(re.findall(r'\*\*([^*]+)\*\*', text)):
        n = re.findall(r'-?\d[\d,\.]*', b)
        if n:
            return _clean(n[-1])
    n = re.findall(r'-?\d[\d,\.]*', text)
    return _clean(n[-1]) if n else None


# ============ کمک‌تابع‌ها ============
def load_questions(path, n):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr); sys.exit(1)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
            if len(data) >= n: break
    return data


def base_url(endpoint):
    return endpoint.rsplit("/v1/", 1)[0] if "/v1/" in endpoint else endpoint.rsplit("/", 1)[0]


def fetch_props(endpoint, timeout=5):
    """خواندن /props برای ثبت پیکربندی واقعی سرور."""
    base = base_url(endpoint)
    for path in ("/props", "/v1/props"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
    return None


# ============ run ============
_interrupted = False
def _sigint(sig, frame):
    global _interrupted
    _interrupted = True
    print("\nSIGINT - ذخیرهٔ نتایج جزئی...", file=sys.stderr)
signal.signal(signal.SIGINT, _sigint)


def cmd_run(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_questions(args.data, args.n)
    print(f"📚 {len(data)} سؤال بارگذاری شد", file=sys.stderr)

    props = fetch_props(args.endpoint)
    print(f"📡 /props: {'OK' if props else 'NOT AVAILABLE (manual)'}",
          file=sys.stderr)

    results, correct = [], 0
    t_start = time.time()
    pbar = tqdm(data, desc=f"GSM8K [{args.label}]", unit="q", file=sys.stderr)

    for i, ex in enumerate(pbar):
        if _interrupted: break
        q, gold_raw = ex["question"], ex["answer"]
        gold = extract_answer(gold_raw)

        t0 = time.time()
        try:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": q}],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            r = requests.post(args.endpoint, json=payload, timeout=args.timeout)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            pred = extract_answer(raw)
        except Exception as e:
            raw, pred = f"<ERROR: {e}>", None
        elapsed = time.time() - t0

        ok = pred is not None and pred == gold
        if ok: correct += 1

        results.append({
            "idx": i, "question": q,
            "gold": gold, "gold_raw": gold_raw,
            "pred": pred, "correct": ok,
            "elapsed_s": round(elapsed, 2), "raw": raw,
        })
        pbar.set_postfix(correct=f"{correct}/{i+1}")

    t_total = time.time() - t_start
    total = len(results)
    acc = (correct / total * 100) if total else 0.0
    avg = t_total / total if total else 0.0

    run = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "endpoint": args.endpoint, "model": args.model,
            "n_requested": args.n, "n_completed": total,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "notes": args.notes or "",
            "server_props": props,
            "server_args_manual": args.server_args or "",
            "extractor_version": "v3",
        },
        "summary": {
            "correct": correct, "total": total,
            "accuracy_pct": round(acc, 2),
            "total_time_s": round(t_total, 1),
            "avg_time_per_q_s": round(avg, 2),
        },
        "results": results,
    }

    out = os.path.join(RESULTS_DIR,
        f"{args.label}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {acc:.1f}% ({correct}/{total})  |  {avg:.1f}s/q", file=sys.stderr)
    print(f"💾 {out}", file=sys.stderr)


# ============ compare ============
def load_runs(labels=None):
    runs = []
    for f in sorted(_glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            continue
        if labels and r.get("label") not in labels: continue
        r["_file"] = os.path.basename(f)
        runs.append(r)
    return runs


def _fmt_config(r):
    c = r["config"]
    props = c.get("server_props") or {}
    gen = props.get("default_generation_settings") or {}
    lines = [f"- model: `{props.get('model_alias') or c.get('model','?')}`"]
    if props.get("build_info"):
        lines.append(f"- build: `{props['build_info']}`")
    if gen:
        parts = [f"{k}={gen[k]}" for k in
                 ("n_ctx", "temperature", "top_k", "top_p", "min_p") if k in gen]
        if parts: lines.append(f"- generation: {' '.join(parts)}")
    if c.get("notes"): lines.append(f"- notes: {c['notes']}")
    if c.get("server_args_manual"):
        lines.append(f"- server_args (manual): `{c['server_args_manual']}`")
    if not props and not c.get("server_args_manual"):
        lines.append("- ⚠ هیچ پیکربندی سروری ثبت نشد")
    return "\n".join(lines)


def to_markdown(runs):
    out = [f"# گزارش مقایسهٔ GSM8K\n",
           f"_تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
           "## خلاصه\n",
           "| Label | زمان | N | دقت | درست | زمان/سؤال | توضیحات |",
           "|-------|------|---|-----|------|-----------|---------|"]
    for r in runs:
        s, c = r["summary"], r["config"]
        out.append(f"| `{r['label']}` | {r['timestamp'][:16]} | {s['total']} "
                   f"| **{s['accuracy_pct']:.1f}%** | {s['correct']}/{s['total']} "
                   f"| {s['avg_time_per_q_s']:.1f}s | {c.get('notes','')} |")

    if len(runs) >= 2:
        n = min(r["summary"]["total"] for r in runs)
        out += ["", f"## موارد اختلاف (اولین {n} سؤال)\n",
                "| idx | gold | " + " | ".join(r["label"] for r in runs) + " |",
                "|-----|------|" + "|".join("---" for _ in runs) + "|"]
        diff = 0
        for i in range(n):
            cor = [r["results"][i]["correct"] if i < len(r["results"]) else None
                   for r in runs]
            if len(set(cor)) <= 1: continue
            diff += 1
            gold = runs[0]["results"][i]["gold"]
            cells = []
            for r in runs:
                if i >= len(r["results"]): cells.append("—"); continue
                res = r["results"][i]
                cells.append(("✓" if res["correct"] else "✗") + f" `{res['pred']}`")
            out.append(f"| {i} | {gold} | " + " | ".join(cells) + " |")
        if diff == 0:
            out.append("| — | — | " + " | ".join("هم‌خوان" for _ in runs) + " |")

    out += ["", "## پیکربندی هر اجرا\n"]
    for r in runs:
        out.append(f"### `{r['label']}`")
        out.append(f"- زمان: {r['timestamp']}")
        out.append(f"- فایل: `{r['_file']}`")
        out.append(f"- استخراج: {r['config'].get('extractor_version','?')}")
        out.append(_fmt_config(r)); out.append("")
    return "\n".join(out)


def cmd_compare(args):
    labels = args.labels.split(",") if args.labels else None
    runs = load_runs(labels)
    if not runs:
        print("❌ هیچ اجرایی پیدا نشد", file=sys.stderr); sys.exit(1)
    md = to_markdown(runs)
    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as f:
            f.write(md)
        print(f"💾 {args.out}", file=sys.stderr)
    else:
        print(md)


def cmd_list(args):
    runs = load_runs()
    if not runs:
        print(f"📂 هیچ اجرایی در {RESULTS_DIR} نیست", file=sys.stderr); return
    print(f"📂 {len(runs)} اجرا در {RESULTS_DIR}:")
    for r in runs:
        s = r["summary"]
        print(f"  {r['label']:25s} {s['accuracy_pct']:6.1f}%  "
              f"({s['correct']}/{s['total']})  {r['timestamp'][:16]}")


# ============ re-extract ============
def cmd_re_extract(args):
    files = sorted(set(sum((_glob.glob(os.path.expanduser(p))
                            for p in args.patterns), [])))
    if not files:
        print("❌ هیچ فایلی مطابقت نداشت", file=sys.stderr); sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            run = json.load(f)
        correct = 0
        for r in run["results"]:
            r["pred"] = extract_answer(r.get("raw", ""))
            if r.get("gold_raw"):
                r["gold"] = extract_answer(r["gold_raw"])
            r["correct"] = r["pred"] is not None and r["pred"] == r["gold"]
            if r["correct"]: correct += 1
        total = len(run["results"])
        acc = (correct / total * 100) if total else 0.0
        run["summary"]["correct"] = correct
        run["summary"]["accuracy_pct"] = round(acc, 2)
        run["config"]["extractor_version"] = "v3"
        run["label"] = args.label or (run["label"] + "__reextract")
        run["timestamp"] = datetime.now().isoformat(timespec="seconds")
        out = os.path.join(RESULTS_DIR,
            f"{run['label']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        print(f"🔄 {os.path.basename(path)} → {acc:.1f}% ({correct}/{total})")
        print(f"   💾 {out}")


# ============ main ============
def main():
    ap = argparse.ArgumentParser(description="GSM8K for llama.cpp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--notes", default="")
    p.add_argument("--server-args", default="")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare")
    p.add_argument("--labels", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("re-extract")
    p.add_argument("patterns", nargs="+")
    p.add_argument("--label", default=None)
    p.set_defaults(func=cmd_re_extract)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()