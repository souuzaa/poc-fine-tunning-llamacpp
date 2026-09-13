#!/usr/bin/env python
"""Stage 6 -- render the base-vs-tuned comparison as a standalone HTML page.

The metrics table answers "did fine-tuning help?"; the three-column comparison is the
guard against trusting a number that is technically true and substantively wrong.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import load_config  # noqa: E402

N_EXAMPLES = 30

STYLE = """
:root { --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e3e3e0;
        --good:#1a7f4b; --bad:#b4341f; --card:#fff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#16161a; --fg:#eceae6; --muted:#9a9a95; --line:#2e2e33; --card:#1e1e23; } }
:root[data-theme=dark] { --bg:#16161a; --fg:#eceae6; --muted:#9a9a95;
  --line:#2e2e33; --card:#1e1e23; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6
  ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1400px; margin:0 auto; padding:40px 24px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
h2 { font-size:20px; margin:40px 0 4px; letter-spacing:-.01em; }
.sub { color:var(--muted); margin:0 0 32px; }
.tablewrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; margin-bottom:40px; min-width:480px; }
th,td { text-align:left; padding:10px 14px; border-bottom:1px solid var(--line); }
th { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.win { color:var(--good); font-weight:600; } .lose { color:var(--bad); }
.ex { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:18px; margin-bottom:20px; }
.attrs { font-size:13px; color:var(--muted); margin-bottom:14px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
.cols { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
@media (max-width:900px) { .cols { grid-template-columns:1fr; } }
.col h4 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); }
.col div.body { font-size:13px; white-space:pre-wrap; max-height:380px;
  overflow-y:auto; border-left:2px solid var(--line); padding-left:12px; }
"""


def metric_row(label: str, base, tuned, lower_is_better: bool = False) -> str:
    def cell(value, other):
        if value is None:
            return '<td class="num">n/a</td>'
        if other is None:
            return f'<td class="num">{value:.4f}</td>'
        better = value < other if lower_is_better else value > other
        css = "win" if better else ("lose" if value != other else "")
        return f'<td class="num {css}">{value:.4f}</td>'

    return f"<tr><td>{html.escape(label)}</td>{cell(base, tuned)}{cell(tuned, base)}</tr>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3-8b-personas.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    eval_dir = Path(cfg.paths.outputs_dir) / "eval"
    results = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    generations = json.loads((eval_dir / "generations.json").read_text(encoding="utf-8"))

    base, tuned = results["models"]["base"], results["models"]["tuned"]
    rows = [
        metric_row("Perplexity (held-out)", base["perplexity"], tuned["perplexity"], True),
        metric_row("Format validity", base["format_validity"], tuned["format_validity"]),
        metric_row("Grounding recall", base["grounding_recall"], tuned["grounding_recall"]),
        metric_row("Think leak rate", base["think_leak_rate"], tuned["think_leak_rate"], True),
        metric_row("Portuguese rate", base["portuguese_rate"], tuned["portuguese_rate"]),
    ]
    for field in ("name", "municipality", "state", "occupation", "age"):
        rows.append(metric_row(
            f"  grounding: {field}",
            base["grounding_by_field"][field],
            tuned["grounding_by_field"][field],
        ))

    examples = []
    for item in generations[:N_EXAMPLES]:
        attrs = item["attributes"]
        header = (f"{attrs['name']} | {attrs['sex']} | {attrs['age']} | "
                  f"{attrs['occupation']} | {attrs['municipality']}, {attrs['state']}")
        columns = "".join(
            f'<div class="col"><h4>{title}</h4>'
            f'<div class="body">{html.escape(item[key])}</div></div>'
            for title, key in (("Ground truth", "reference"),
                               ("Base", "base"), ("Fine-tuned", "tuned"))
        )
        examples.append(
            f'<div class="ex"><div class="attrs">{html.escape(header)}</div>'
            f'<div class="cols">{columns}</div></div>'
        )

    page = f"""<title>Persona Fine-Tune Results</title>
<style>{STYLE}</style>
<div class="wrap">
<h1>Qwen3-8B persona fine-tune &mdash; base vs tuned</h1>
<p class="sub">{results['n_rows']} held-out rows &middot; prompt version
{html.escape(results['prompt_version'])} &middot; Q4_K_M on RTX 3060 &middot; identical
decoding parameters</p>
<div class="tablewrap">
<table>
<thead><tr><th>Metric</th><th style="text-align:right">Base</th>
<th style="text-align:right">Fine-tuned</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
<h2>Side-by-side generations</h2>
<p class="sub">First {min(N_EXAMPLES, len(generations))} held-out attribute sets.</p>
{''.join(examples)}
</div>"""

    out = eval_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f">> wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
