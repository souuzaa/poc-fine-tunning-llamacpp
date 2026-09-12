#!/usr/bin/env python
"""Stage 1 -- build train/val/test JSONL from nvidia/Nemotron-Personas-Brazil.

Streams the parquet so the 2.5GB corpus is never fully materialised. For every row:
extract the name (ADR 0005), render the prompt and target through the shared renderer
(ADR 0006), measure token length, and keep it only if it fits max_seq_length.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402
from personas.names import extract_name  # noqa: E402
from personas.prompt import (  # noqa: E402
    PROMPT_VERSION,
    render_prompt,
    render_training_text,
)
from personas.schema import (  # noqa: E402
    NAME_SOURCE_COLUMNS,
    SECTIONS,
    PersonaAttributes,
)


def name_for(row: dict) -> str | None:
    """Extract the name, falling back to the other narrative columns (ADR 0005)."""
    return extract_name(
        str(row.get("persona") or ""),
        [str(row.get(column) or "") for column in NAME_SOURCE_COLUMNS],
    )


def build_record(row: dict, tokenizer) -> dict | None:
    """Turn a source row into a training record, or None if it is unusable."""
    name = name_for(row)
    if name is None:
        return None
    for _, column in SECTIONS:
        if not str(row.get(column) or "").strip():
            return None

    attrs = PersonaAttributes.from_row(row, name)
    prompt = render_prompt(attrs, tokenizer)
    text = render_training_text(attrs, row, tokenizer)
    if not text.startswith(prompt):
        raise RuntimeError(
            "prompt/training-text drift detected -- see ADR 0006 and "
            "tests/test_prompt_parity.py"
        )
    return {
        "uuid": row["uuid"],
        "attributes": attrs.__dict__,
        "prompt": prompt,
        "target": text[len(prompt):],
        "text": text,
        "n_tokens": len(tokenizer(text)["input_ids"]),
    }


def split_indices(total: int, n_train: int, n_val: int, n_test: int, seed: int):
    """Disjoint index splits over `total` usable records."""
    needed = n_train + n_val + n_test
    if total < needed:
        raise ValueError(f"only {total} usable rows, need {needed}")
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return (
        order[:n_train],
        order[n_train:n_train + n_val],
        order[n_train + n_val:needed],
    )


def histogram(lengths: list[int]) -> str:
    if not lengths:
        return "(no records)"
    ordered = sorted(lengths)

    def pct(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

    buckets = Counter(min(length // 256 * 256, 3072) for length in lengths)
    lines = [
        f"  n={len(lengths)} min={ordered[0]} p50={pct(0.5)} "
        f"p90={pct(0.9)} p99={pct(0.99)} max={ordered[-1]}"
    ]
    for bucket in sorted(buckets):
        bar = "#" * max(1, round(40 * buckets[bucket] / len(lengths)))
        lines.append(f"  {bucket:>5}+ {bar} {buckets[bucket]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny splits for an end-to-end pipeline check")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_train, n_val, n_test = cfg.data.n_train, cfg.data.n_val, cfg.data.n_test
    if args.smoke:
        n_train, n_val, n_test = 160, 20, 20

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    stream = load_dataset(cfg.data.hf_dataset, split="train", streaming=True)

    # Oversample: some rows are dropped for a missing name or an over-long target.
    target_count = n_train + n_val + n_test
    scan_limit = int(target_count * 1.6) + 500

    records, seen, dropped_name, dropped_empty, dropped_long = [], 0, 0, 0, 0
    lengths: list[int] = []

    for row in stream:
        seen += 1
        if seen > scan_limit or len(records) >= target_count:
            break
        if name_for(row) is None:
            dropped_name += 1
            continue
        record = build_record(row, tokenizer)
        if record is None:
            dropped_empty += 1
            continue
        lengths.append(record["n_tokens"])
        if record["n_tokens"] > cfg.model.max_seq_length:
            dropped_long += 1
            continue
        records.append(record)

    drop_rate = dropped_name / max(1, seen)
    print(f"scanned={seen} usable={len(records)} "
          f"dropped: name={dropped_name} ({drop_rate:.1%}) "
          f"empty={dropped_empty} too_long={dropped_long}")
    print("token length distribution (full training text):")
    print(histogram(lengths))

    if drop_rate > cfg.data.max_name_drop_rate:
        print(f"\nFATAL: name extraction failed on {drop_rate:.1%} of rows, "
              f"limit is {cfg.data.max_name_drop_rate:.1%}. "
              f"Fix the regex in src/personas/names.py (ADR 0005).", file=sys.stderr)
        return 1

    train_idx, val_idx, test_idx = split_indices(
        len(records), n_train, n_val, n_test, cfg.data.seed
    )

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        path = data_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in indices:
                handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(indices)} rows)")

    uuids = {name: {records[i]["uuid"] for i in idx}
             for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))}
    assert not (uuids["train"] & uuids["val"]), "train/val overlap"
    assert not (uuids["train"] & uuids["test"]), "train/test overlap"
    assert not (uuids["val"] & uuids["test"]), "val/test overlap"

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "base_id": cfg.model.base_id,
        "dataset": cfg.data.hf_dataset,
        "max_seq_length": cfg.model.max_seq_length,
        "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "scanned": seen,
        "dropped": {"name": dropped_name, "empty": dropped_empty, "too_long": dropped_long},
        "smoke": args.smoke,
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {data_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    code = main()
    # `datasets` streaming leaves a reader thread that aborts the process during
    # interpreter finalization ("PyGILState_Release: thread state must be current"),
    # turning a successful run into exit code 134. It reproduces in four lines with
    # none of our code, and explicit cleanup does not help. Every output is already
    # flushed to disk by this point, so skipping finalization costs nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
