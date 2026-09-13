#!/usr/bin/env python
"""Stage 3 -- merge, convert and quantise (ADR 0008).

Three explicit steps rather than Unsloth's one-call GGUF helper: each fails in isolation
with a legible error and can be re-run without repeating the others.

The base model goes through the IDENTICAL chain so that base-vs-tuned differs only by the
LoRA weights, not by quantisation lineage or converter version.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402

QUANTS = ("Q4_K_M", "Q8_0")


def run(command: list[str]) -> None:
    print(">>", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def merge_tuned(cfg, merged_dir: Path) -> None:
    """Merge the LoRA adapter into fp16 weights. Peak RAM ~17GB."""
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    adapter = Path(cfg.paths.outputs_dir) / "adapter"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


def materialise_base(cfg, merged_dir: Path) -> None:
    """Write the untuned base in fp16 through the same save path as the merge."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_id, dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_id)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    parser.add_argument("--which", choices=("tuned", "base"), required=True)
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    outputs = Path(cfg.paths.outputs_dir)
    llamacpp = Path(cfg.paths.llamacpp_dir)
    gguf_dir = outputs / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = outputs / f"merged-16bit-{args.which}"
    f16_path = gguf_dir / f"personas-{args.which}-f16.gguf"

    # Step 1 -- fp16 weights on disk.
    # Reuse is for resumability, but a merge older than the adapter it came from is
    # STALE: retraining and re-exporting would silently ship the previous model and
    # invalidate every number in the evaluation. Detect it and re-merge.
    if args.which == "tuned" and (merged_dir / "config.json").exists():
        adapter_weights = Path(cfg.paths.outputs_dir) / "adapter" / "adapter_model.safetensors"
        if (adapter_weights.exists()
                and adapter_weights.stat().st_mtime > (merged_dir / "config.json").stat().st_mtime):
            print(f">> {merged_dir} is older than the adapter -- discarding stale merge")
            shutil.rmtree(merged_dir)
            f16_path.unlink(missing_ok=True)
            for quant in QUANTS:
                (gguf_dir / f"personas-tuned-{quant.lower()}.gguf").unlink(missing_ok=True)

    if not (merged_dir / "config.json").exists():
        print(f">> materialising fp16 weights for '{args.which}' (~16GB, RAM peak ~17GB)")
        if args.which == "tuned":
            merge_tuned(cfg, merged_dir)
        else:
            materialise_base(cfg, merged_dir)
    else:
        print(f">> reusing {merged_dir}")

    # Step 2 -- HF to GGUF f16
    if not f16_path.exists():
        run([
            sys.executable,
            str(llamacpp / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ])
    else:
        print(f">> reusing {f16_path}")

    # Step 3 -- quantise
    quantize = llamacpp / "build" / "bin" / "llama-quantize"
    for quant in QUANTS:
        out = gguf_dir / f"personas-{args.which}-{quant.lower()}.gguf"
        if out.exists():
            print(f">> reusing {out}")
            continue
        run([str(quantize), str(f16_path), str(out), quant])
        print(f">> {out.name}: {out.stat().st_size / 1024**3:.2f} GB")

    if not args.keep_intermediates:
        print(">> removing intermediates (re-runnable; pass --keep-intermediates to keep)")
        f16_path.unlink(missing_ok=True)

    print(">> export complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
