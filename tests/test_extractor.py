def test_real_qwen_outputs_accuracy(qwen_outputs):
    """Reference: 18/20 on the committed fixture (see docs/extraction-bug.md)."""
    assert len(qwen_outputs) == 20, (
        f"fixture has {len(qwen_outputs)} rows, expected 20"
    )
    correct = sum(
        1 for row in qwen_outputs
        if extract_answer(row["raw"]) == row["gold"]
    )
    assert correct == 18, (
        f"expected 18/20, got {correct}/20\n"
        f"If you changed the extractor, update docs/extraction-bug.md "
        f"and this assertion together."
    )


def test_real_qwen_outputs_known_failures(qwen_outputs):
    """The two remaining failures are real model errors, not extraction bugs."""
    by_idx = {row["idx"]: row for row in qwen_outputs}
    known = {7: ("160", "100"), 12: ("13", "12")}
    for idx, (gold, pred) in known.items():
        row = by_idx[idx]
        assert row["gold"] == gold, f"idx={idx}: gold mismatch"
        assert extract_answer(row["raw"]) == pred, f"idx={idx}: pred mismatch"