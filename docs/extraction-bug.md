# The extraction bug: how the same model scored 62%, 35%, and 90%

## TL;DR

A single Qwen3.5-4B Q4_K_M model, on the same 20 GSM8K questions,
produced three very different accuracy scores depending only on how
the final answer was extracted from the model's text:

| Extractor version | Accuracy | What went wrong |
|---|---|---|
| v1: only `####` + last number | **62%** | Missed `**bold**` final answers |
| v2: first `=` pattern | **35%** | Matched the first `=` in the working, not the answer |
| v3: `####` → `\boxed{}` → `**bold**` → last number | **90%** | Aligned with the model's actual output format |

The 62% and 35% numbers were **empty numbers** — they measured the
extractor, not the model.

## The setup

- Model: `Qwen3.5-4B-super-coder.Q4_K_M.gguf`
- Server: `llama-server` on CPU (8 threads, Termux / Windows)
- Data: first 20 rows of GSM8K test set (`openai/grade-school-math`)

## v1: only `####` and last number (62%)

The first extractor was:

```python
def extract_answer(text):
    m = re.search(r'####\s*(-?\d[\d,\.]*)', text)
    if m:
        return m.group(1)
    nums = re.findall(r'-?\d[\d,\.]*', text)
    return nums[-1] if nums else None
```

Qwen3.5 does not emit `####`. It writes the final answer in **bold**.
Every question therefore fell through to "last number", which frequently
was not the answer.

### Failing examples

| idx | gold | last number | why wrong |
|---|---|---|---|
| 4 | 20 | **60** | trailing `Let me verify: ... = 60 cups` |
| 8 | 45 | **4** | trailing `at the end of those 4 hours` |
| 15 | 125 | **1.2** | trailing `(2.5% vs 1.2%)` |
| 18 | 7 | **4** | trailing `in 4 weeks` |

## v2: first `=` pattern (35%)

A naïve attempt to fix v1:

```python
m = re.search(r'=\s*\$?\s*(-?\d[\d,\.]*)', text)
if m:
    return m.group(1)
```

This matches the **first** `=` in the working, not the final one.

### Failing examples

| idx | gold | first `=` value | why wrong |
|---|---|---|---|
| 2 | 70000 | **150** | `Increase = 150% × $80,000` |
| 9 | 460 | **5** | `45 - 40 = 5 hours` |
| 17 | 57500 | **1750** | `50 weeks × 35 hours = 1,750 hours` |

This was a **regression**: v2 was worse than v1.

## v3: aligned with the model's output format (90%)

The fix was not a better regex — it was **reading the actual output**:

```python
# 1. #### 42
m = re.search(r'####\s*(-?\d[\d,\.]*)', text)
if m:
    return _clean(m.group(1))

# 2. \boxed{42}  (last occurrence, tolerate \text{...})
for box in reversed(re.findall(r'\\boxed\s*\{([^}]*)\}', text)):
    nums = re.findall(r'-?\d[\d,\.]*', box)
    if nums:
        return _clean(nums[-1])

# 3. **42**       (last occurrence)
for bold in reversed(re.findall(r'\*\*([^*]+)\*\*', text)):
    nums = re.findall(r'-?\d[\d,\.]*', bold)
    if nums:
        return _clean(nums[-1])

# 4. last number  (fallback)
nums = re.findall(r'-?\d[\d,\.]*', text)
return _clean(nums[-1]) if nums else None
```

Qwen3.5 consistently wraps its final answer in `**...**`. Taking the
**last bold span** fixed 5 of the 6 failures.

### The two remaining failures

| idx | gold | pred | verdict |
|---|---|---|---|
| 7 | 160 | 100 | Model computed `60 + 60 = 100` (arithmetic error) |
| 12 | 13 | 12 | Model computed `90 ÷ 7.5 = 12` (correct math, disagrees with gold) |

Both are **real model errors**, not extraction bugs. The extractor
correctly reads `100` and `12` from the bold spans.

## Lessons

1. **The extractor is part of the measurement tool, not a neutral reader.**
   It needs its own validation on real outputs.

2. **"Official" formats are not universal.** The GSM8K dataset uses `####`,
   but instruct-tuned models use `\boxed{}` or `**bold**`. A toolkit that
   only handles the dataset's format will miss the model's format.

3. **Regressions are invisible without a fixed test set.** v2 looked like
   an improvement but was worse. Only a stable 20-question fixture caught
   it.

4. **The "empty number" trap has three layers**: engine, config, and
   measurement. This project addresses the measurement layer.

5. **A benchmark is only as good as its extractor.** Two teams running
   the same model on the same questions can report wildly different
   accuracy scores simply because their extractors differ.
```

---

## 📄 پچ ۱۰: `Makefile` (جایگزین کامل)

```makefile
.PHONY: install test test-strict lint clean build publish

install:
	pip install -e ".[dev]"

test:
	pytest

test-strict:
	LLM_QUALITY_ALLOW_MISSING_FIXTURE=0 pytest

lint:
	ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*


