"""Internal utilities for llm-quality-toolkit."""

from __future__ import annotations

import re

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]+")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")


def redact_sensitive(text: str | None) -> str | None:
    """Remove Bearer tokens and ``sk-*`` keys from a string.

    Used anywhere an error message or URL might be logged.
    """
    if not text:
        return text
    text = _BEARER.sub(r"\1<REDACTED>", text)
    text = _SK_KEY.sub("<REDACTED>", text)
    return text