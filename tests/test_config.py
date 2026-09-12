from pathlib import Path

from personas.config import load_config

CONFIG = Path(__file__).parent.parent / "configs" / "qwen3-8b-personas.yaml"


def test_load_config_exposes_nested_attributes():
    cfg = load_config(CONFIG)
    assert cfg.model.base_id == "unsloth/Qwen3-8B"
    assert cfg.model.max_seq_length == 2048
    assert cfg.lora.r == 32
    assert cfg.train.per_device_train_batch_size == 2
    assert cfg.decode.top_k == 20


def test_effective_batch_size_is_sixteen():
    cfg = load_config(CONFIG)
    effective = cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps
    assert effective == 16


def test_splits_are_the_budget_from_adr_0009():
    cfg = load_config(CONFIG)
    assert (cfg.data.n_train, cfg.data.n_val, cfg.data.n_test) == (20000, 500, 200)
