CONFIG ?= configs/qwen3-8b-personas.yaml
UV     ?= uv

.PHONY: setup data train export serve eval report smoke test clean

setup:
	$(UV) sync --extra train --extra eval --extra dev
	$(UV) run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
	bash scripts/00_setup_llamacpp.sh

data:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG)

train:
	$(UV) run python scripts/02_train.py --config $(CONFIG)

export:
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which tuned
	$(UV) run python scripts/03_export_gguf.py --config $(CONFIG) --which base

serve:
	bash scripts/04_serve.sh $(MODEL)

eval:
	$(UV) run python scripts/05_evaluate.py --config $(CONFIG)

report:
	$(UV) run python scripts/06_report.py --config $(CONFIG)

test:
	$(UV) run --extra dev pytest -v

smoke:
	$(UV) run python scripts/01_prepare_data.py --config $(CONFIG) --smoke
	@echo "Smoke data ready. Run: make train export eval report"

clean:
	rm -rf outputs/merged-16bit-* outputs/gguf/*-f16.gguf
