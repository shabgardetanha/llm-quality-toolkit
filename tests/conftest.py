"""Pytest fixtures."""

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def qwen_outputs() -> list[dict]:
    """20 real Qwen3.5-4B outputs committed to the repo.

    Regenerate with: ``python3 scripts/make_fixture.py --latest``

    If the fixture is missing, tests FAIL (not skip) so CI catches it.
    Set ``LLM_QUALITY_ALLOW_MISSING_FIXTURE=1`` to skip instead.
    """
    path = FIXTURES / "qwen_outputs.jsonl"
    if not path.exists():
        if os.environ.get("LLM_QUALITY_ALLOW_MISSING_FIXTURE") == "1":
            pytest.skip(f"fixture missing (allowed by env var): {path}")
        pytest.fail(
            f"fixture missing: {path}\n"
            f"  This file should be committed to the repository.\n"
            f"  Regenerate: python3 scripts/make_fixture.py --latest\n"
            f"  Or skip:    LLM_QUALITY_ALLOW_MISSING_FIXTURE=1 pytest"
        )
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]