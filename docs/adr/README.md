# Architecture Decision Records

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-qlora-not-full-finetune.md) | 4-bit QLoRA, not a full fine-tune | Accepted |
| [0003](0003-base-model-qwen3-8b.md) | Base model: Qwen3-8B | Accepted |
| [0004](0004-task-attributes-to-persona.md) | Task: attributes to persona narrative | Accepted |
| [0005](0005-name-and-facet-scope.md) | Inject name as input; drop lifestyle facets | Accepted |
| [0006](0006-prompt-parity-and-thinking-mode.md) | One shared prompt renderer; thinking mode off | Accepted |
| [0007](0007-langchain-scope.md) | LangChain scoped to eval and serving | Accepted |
| [0008](0008-gguf-export-path.md) | Explicit merge, convert, quantise for GGUF | Accepted |
| [0009](0009-training-budget.md) | 20k examples, one epoch | Superseded by 0012 |
| [0010](0010-evaluation-strategy.md) | Automatic metrics plus side-by-side | Accepted |
| [0011](0011-python-env-and-pinning.md) | Isolated uv env on Python 3.12, pinned | Accepted |
| [0012](0012-revised-training-budget.md) | Revised budget: 10k examples (measured throughput) | Accepted |

New decisions get the next number. ADRs are immutable once Accepted — a reversal is a new
ADR that supersedes the old one.
