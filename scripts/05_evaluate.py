#!/usr/bin/env python
"""Stage 5 -- generate over the test split with base and tuned, then score (ADR 0010).

Runs the two models SEQUENTIALLY. Two Q4_K_M 8B models do not fit in ~9.5GB of free
VRAM, so this script owns server lifecycle: start, wait, generate, tear down, repeat.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402
from personas.grounding import check_format, is_portuguese, score_grounding  # noqa: E402
from personas.llm import LlamaCppCompletion, wait_for_server  # noqa: E402
from personas.prompt import PROMPT_VERSION  # noqa: E402

PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"


def load_test_rows(data_dir: Path) -> list[dict]:
    with (data_dir / "test.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def generate_with(model_path: Path, prompts: list[str], cfg, concurrency: int = 2):
    """Start llama-server, generate, always tear it down."""
    print(f">> starting llama-server for {model_path.name}")
    server = subprocess.Popen(
        ["bash", "scripts/04_serve.sh", str(model_path), str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(BASE_URL)
        print(f">> ready; generating {len(prompts)} completions")
        llm = LlamaCppCompletion(BASE_URL, cfg.decode)
        started = time.time()
        outputs = llm.batch(prompts, config={"max_concurrency": concurrency})
        print(f">> done in {time.time() - started:.0f}s")
        llm.close()
        return outputs
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        print(">> server stopped")
        time.sleep(5)  # let the driver release VRAM before the next model loads


def score(rows: list[dict], generations: list[str]) -> dict:
    per_row = []
    for row, generated in zip(rows, generations):
        fmt = check_format(generated)
        grounding = score_grounding(generated, row["attributes"])
        per_row.append({
            "uuid": row["uuid"],
            "format": fmt,
            "grounding": grounding,
            "portuguese": is_portuguese(generated),
            "chars": len(generated),
        })
    return {
        "format_validity": mean(r["format"]["valid"] for r in per_row),
        "think_leak_rate": mean(r["format"]["think_leak"] for r in per_row),
        "grounding_recall": mean(r["grounding"]["recall"] for r in per_row),
        "grounding_by_field": {
            field: mean(r["grounding"][field] for r in per_row)
            for field in ("name", "municipality", "state", "occupation", "age")
        },
        "portuguese_rate": mean(r["portuguese"] for r in per_row),
        "mean_chars": mean(r["chars"] for r in per_row),
        "per_row": per_row,
    }


def perplexity(model_path: Path, corpus: Path, cfg) -> float | None:
    """Held-out perplexity via llama-perplexity. Returns None if it fails."""
    binary = Path(cfg.paths.llamacpp_dir) / "build" / "bin" / "llama-perplexity"
    try:
        result = subprocess.run(
            [str(binary), "-m", str(model_path), "-f", str(corpus),
             "-ngl", "99", "-c", "2048", "--chunks", "40"],
            capture_output=True, text=True, timeout=3600, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f">> perplexity failed for {model_path.name}: {error}", file=sys.stderr)
        return None
    matches = re.findall(r"Final estimate: PPL = ([\d.]+)", result.stdout + result.stderr)
    return float(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.paths.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != PROMPT_VERSION:
        print(f"FATAL: prompt version mismatch -- data {manifest['prompt_version']}, "
              f"code {PROMPT_VERSION} (ADR 0006)", file=sys.stderr)
        return 1

    rows = load_test_rows(data_dir)
    if args.limit:
        rows = rows[:args.limit]
    prompts = [row["prompt"] for row in rows]
    print(f">> evaluating on {len(rows)} held-out rows")

    gguf_dir = Path(cfg.paths.outputs_dir) / "gguf"
    models = {
        "base": gguf_dir / "personas-base-q4_k_m.gguf",
        "tuned": gguf_dir / "personas-tuned-q4_k_m.gguf",
    }
    for name, path in models.items():
        if not path.exists():
            print(f"FATAL: {name} model missing at {path}. Run `make export`.",
                  file=sys.stderr)
            return 1

    # Reference corpus for perplexity: the held-out targets.
    eval_dir = Path(cfg.paths.outputs_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    corpus = eval_dir / "test_corpus.txt"
    corpus.write_text("\n\n".join(row["target"] for row in rows), encoding="utf-8")

    results: dict = {"prompt_version": PROMPT_VERSION, "n_rows": len(rows), "models": {}}
    generations: dict[str, list[str]] = {}

    for name, path in models.items():  # sequential -- see module docstring
        outputs = generate_with(path, prompts, cfg)
        generations[name] = outputs
        results["models"][name] = score(rows, outputs)
        results["models"][name]["perplexity"] = perplexity(path, corpus, cfg)

    (eval_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (eval_dir / "generations.json").write_text(
        json.dumps(
            [
                {
                    "uuid": row["uuid"],
                    "attributes": row["attributes"],
                    "reference": row["target"],
                    "base": generations["base"][i],
                    "tuned": generations["tuned"][i],
                }
                for i, row in enumerate(rows)
            ],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    header = f"{'metric':<24}{'base':>12}{'tuned':>12}"
    print(header)
    print("-" * len(header))

    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    for metric in ("perplexity", "format_validity", "grounding_recall",
                   "think_leak_rate", "portuguese_rate", "mean_chars"):
        base_value = results["models"]["base"][metric]
        tuned_value = results["models"]["tuned"][metric]
        print(f"{metric:<24}{fmt(base_value):>12}{fmt(tuned_value):>12}")
    print(f"\nwrote {eval_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
