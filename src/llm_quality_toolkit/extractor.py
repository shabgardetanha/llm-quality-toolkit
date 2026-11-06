"""Answer extraction for GSM8K with multi-format support.

Priority (validated on Qwen3.5-4B real outputs):

1. ``#### 42``      — official GSM8K format
2. ``\\boxed{42}``  — common in math-tuned models
3. ``**42**``       — Qwen instruct models use bold for the final answer
4. last number      — fallback

The design rationale is documented in ``docs/extraction-bug.md``.
"""

from __future__ import annotations

import re
from typing import Optional

_NUM = r"-?\d[\d,\.]*"
_RE_HASH = re.compile(rf"####\s*({_NUM})")
_RE_BOXED = re.compile(r"\\boxed\s*\{([^}]*)\}")
_RE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_RE_NUM = re.compile(_NUM)


def _clean(s: str) -> str:
    """Strip commas, trailing periods, and whitespace."""
    return s.replace(",", "").rstrip(".").strip()


def extract_answer(text: Optional[str]) -> Optional[str]:
    """Extract the final numeric answer from a model response.

    Returns ``None`` if no number can be found.
    """
    if not text:
        return None

    # 1. #### 42 (official GSM8K)
    m = _RE_HASH.search(text)
    if m:
        return _clean(m.group(1))

    # 2. \boxed{...} — last occurrence, tolerate \text{...} inside
    for box in reversed(_RE_BOXED.findall(text)):
        nums = _RE_NUM.findall(box)
        if nums:
            return _clean(nums[-1])

    # 3. **bold** — last occurrence, take last number in the bold span
    for bold in reversed(_RE_BOLD.findall(text)):
        nums = _RE_NUM.findall(bold)
        if nums:
            return _clean(nums[-1])

    # 4. last number anywhere
    nums = _RE_NUM.findall(text)
    return _clean(nums[-1]) if nums else None