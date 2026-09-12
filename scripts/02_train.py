#!/usr/bin/env python
"""Stage 2 -- QLoRA fine-tune Qwen3-8B on a single RTX 3060 (ADR 0002, 0009).

Trains on the assistant response only, so the loss never rewards reproducing the prompt.
Prints a measured ETA after a short warmup: a wrong throughput estimate should surface in
minutes, not at hour four.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# torch inductor spawns one compile worker per CPU core -- 16 separate Python processes
# on this machine -- when kernels are JIT-compiled at the first training step. That spike
# can exhaust system RAM before training has allocated anything real, killing the run
# minutes in. Cap it; compilation happens once and a few seconds slower costs nothing
# against a multi-hour run. Must be set before torch is imported.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")
# Training sits at ~8GB of ~9.5GB free, so the allocator has little room to manoeuvre;
# expandable segments avoid fragmentation stalling an otherwise-fitting allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: E402,F401  # must precede transformers/trl -- it patches on import
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402
from personas.prompt import PROMPT_VERSION  # noqa: E402

WARMUP_STEPS_FOR_ETA = 50


def make_eta_callback(total_steps: int):
    """A TrainerCallback that reports a measured ETA once throughput is known."""
    from transformers import TrainerCallback

    class EtaCallback(TrainerCallback):
        def __init__(self):
            self.start: float | None = None
            self.reported = False

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if step == 1:
                self.start = time.time()
            elif (step >= WARMUP_STEPS_FOR_ETA and self.start is not None
                  and not self.reported):
                per_step = (time.time() - self.start) / (step - 1)
                remaining = (total_steps - step) * per_step
                print(f"\n>> measured {per_step:.2f}s/step -> "
                      f"ETA {remaining / 3600:.1f}h for {total_steps} steps\n",
                      flush=True)
                self.reported = True

    return EtaCallback()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg.paths.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != PROMPT_VERSION:
        print(f"FATAL: data was prepared with prompt version "
              f"{manifest['prompt_version']}, code is at {PROMPT_VERSION}. "
              f"Re-run `make data` (ADR 0006).", file=sys.stderr)
        return 1

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.base_id,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=None,  # auto-detect: bf16 on Ampere
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
    )

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    splits = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    outputs = Path(cfg.paths.outputs_dir)
    (outputs / "adapter").mkdir(parents=True, exist_ok=True)

    effective_batch = (cfg.train.per_device_train_batch_size
                       * cfg.train.gradient_accumulation_steps)
    total_steps = max(
        1,
        (len(splits["train"]) // effective_batch) * cfg.train.num_train_epochs,
    )
    print(f">> {len(splits['train'])} train rows, ~{total_steps} optimizer steps")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        args=SFTConfig(
            output_dir=str(outputs / "checkpoints"),
            dataset_text_field="text",
            max_length=cfg.model.max_seq_length,
            packing=False,  # response-only loss requires unpacked sequences
            per_device_train_batch_size=cfg.train.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
            # Keep only the loss from eval. Without this the trainer accumulates logits
            # for the whole eval set -- vocab 151936 wide -- and OOMs a GPU that just
            # finished training fine.
            prediction_loss_only=True,
            group_by_length=cfg.train.group_by_length,
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            num_train_epochs=cfg.train.num_train_epochs,
            learning_rate=cfg.train.learning_rate,
            lr_scheduler_type=cfg.train.lr_scheduler_type,
            warmup_ratio=cfg.train.warmup_ratio,
            optim=cfg.train.optim,
            weight_decay=cfg.train.weight_decay,
            bf16=True,
            fp16=False,
            logging_steps=cfg.train.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.train.eval_steps,
            save_steps=cfg.train.save_steps,
            save_total_limit=cfg.train.save_total_limit,
            seed=cfg.train.seed,
            report_to="none",
        ),
    )

    # Loss on the assistant turn only. These markers are Qwen3's ChatML delimiters.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.add_callback(make_eta_callback(total_steps))

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f">> finished in {elapsed / 3600:.2f}h, peak VRAM {peak_gb:.2f} GB")

    model.save_pretrained(str(outputs / "adapter"))
    tokenizer.save_pretrained(str(outputs / "adapter"))

    (outputs / "train_log.json").write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "base_id": cfg.model.base_id,
                "train_rows": len(splits["train"]),
                "total_steps": total_steps,
                "train_runtime_hours": elapsed / 3600,
                "peak_vram_gb": peak_gb,
                "final_train_loss": result.training_loss,
                "log_history": trainer.state.log_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f">> adapter saved to {outputs / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
