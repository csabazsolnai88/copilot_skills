# MetricX — Neural Reference-Based & QE Metric

[MetricX](https://github.com/google-research/metricx) is a neural MT quality metric from Google based on mT5.
It outputs a score in **[0, 25]** where **lower raw scores = higher quality**.
The example below swaps the scale so that **higher scores = better**, consistent with BLEU/chrF.

Two model variants exist on HuggingFace:
- `google/metricx-24-hybrid-xxl-v2p6-bfloat16` — **reference-based** (needs `trg`).
- `google/metricx-24-hybrid-xxl-v2p6-bfloat16-qe` — **quality estimation / reference-free** (needs only `src` and `mt`).

---

## Newline splitting helper

Neural metrics have a max-token limit (~1 024 tokens). Paragraphs separated
by newlines are a reliable alignment boundary, so we split them into separate
rows. When the number of newlines differs between hypothesis and
reference/source (i.e. an overgeneration or undergeneration case), we fall
back to collapsing newlines to spaces to avoid index mismatches.

```python
def _split_at_newlines(
    src_segments: list[str],
    trg_segments: list[str],
    mt_segments: list[str],
    is_qe: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Split multi-line segments at newline boundaries before neural scoring."""
    out_src, out_trg, out_mt = [], [], []
    for s, t, m in zip(src_segments, trg_segments, mt_segments):
        s_parts = s.split("\n")
        t_parts = t.split("\n")
        m_parts = m.split("\n")
        aligned = len(m_parts) == len(s_parts) and (is_qe or len(m_parts) == len(t_parts))
        if aligned:
            out_src.extend(s_parts)
            out_trg.extend(t_parts)
            out_mt.extend(m_parts)
        else:
            out_src.append(s.replace("\n", " "))
            out_trg.append(t.replace("\n", " "))
            out_mt.append(m.replace("\n", " "))
    return out_src, out_trg, out_mt
```

## MetricX scoring function

```python
# pip install transformers datasets torch
from __future__ import annotations

import torch
import transformers
from datasets import Dataset
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorWithPadding

import pandas as pd


# ---------------------------------------------------------------------------
# Minimal copy of the MT5ForRegression head (from google-research/metricx).
# If you have fiumicino available, import from there instead:
#   from fiumicino.automated.evaluation.model_metricx import MT5ForRegression, get_dataset
# ---------------------------------------------------------------------------
from fiumicino.automated.evaluation.model_metricx import MT5ForRegression, get_dataset


def _build_metricx_dataset(
    src_segments: list[str],
    trg_segments: list[str],
    mt_segments: list[str],
    tokenizer,
    device: int | str,
    is_qe: bool,
    max_input_length: int = 1024,
):
    """Tokenise triples for MetricX inference."""
    data = [
        {"source": s, "hypothesis": m, "reference": t}
        for s, t, m in zip(src_segments, trg_segments, mt_segments)
    ]
    return get_dataset(data, tokenizer, max_input_length, device, is_qe)


def run_metricx(
    src_segments: list[str],
    trg_segments: list[str],
    mt_segments: list[str],
    output_path: str,
    model_name: str = "google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    tokenizer_name: str = "google/mt5-xl",
    batch_size: int = 8,
    device: int | str = 0,
    swap_scale: bool = True,
) -> pd.DataFrame:
    """Score translation triples with MetricX and write results to output_path.

    Args:
        src_segments: Source segments.
        trg_segments: Reference translations (ignored when using a QE model).
        mt_segments:  Machine translations to score.
        output_path:  Destination CSV/Parquet/XLSX path.
        model_name:   HuggingFace model id.  Append ``-qe`` for reference-free mode.
        tokenizer_name: HuggingFace tokenizer id (default: ``google/mt5-xl``).
        batch_size:   Per-device eval batch size.
        device:       GPU index (int) or ``"cpu"``.
        swap_scale:   If True (default), negate raw scores so higher = better,
                      matching the convention used by BLEU / chrF.

    Returns:
        DataFrame with columns ``src``, ``trg``, ``mt``, ``metricx``.
    """
    is_qe = model_name.endswith("-qe")

    # Split multi-line segments at newline boundaries before tokenisation.
    src_split, trg_split, mt_split = _split_at_newlines(
        src_segments, trg_segments, mt_segments, is_qe
    )

    model = MT5ForRegression.from_pretrained(model_name, torch_dtype="auto")
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, cleanup_tokenization_spaces=False
    )

    data_collator, ds = _build_metricx_dataset(
        src_split, trg_split, mt_split, tokenizer, device, is_qe
    )

    training_args = transformers.TrainingArguments(
        output_dir="dummy_metricx",
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, data_collator=data_collator
    )
    predictions, _, _ = trainer.predict(test_dataset=ds)

    # Raw MetricX scores: 0 = perfect, 25 = worst.
    # Swap so that higher = better (consistent with BLEU/chrF).
    scores = [-float(p) if swap_scale else float(p) for p in predictions]

    df = pd.DataFrame({
        "src": src_segments,
        "trg": trg_segments,
        "mt": mt_segments,
        "metricx": scores,
    })
    write_tabular(df, output_path)
    return df
```

## Usage examples

```python
df = run_metricx(
    src_segments, trg_segments, mt_segments,
    output_path="results.csv",
    model_name="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    device=0,          # GPU index; use "cpu" for CPU-only
    swap_scale=True,   # higher score = better quality
)
print(df["metricx"].describe())

# Reference-free (QE) mode — no reference translation required:
df_qe = run_metricx(
    src_segments, [""] * len(src_segments), mt_segments,
    output_path="results_qe.csv",
    model_name="google/metricx-24-hybrid-xxl-v2p6-bfloat16-qe",
)
```

**Key notes:**
- Raw MetricX scores are in `[0, 25]` with 0 = perfect. `swap_scale=True` negates them so the column behaves like BLEU (larger = better).
- Segments with mismatched newline counts between MT and reference are collapsed to single lines before scoring to avoid alignment errors.
- QE variant (`-qe` suffix) needs only `src` and `mt`; pass empty strings for `trg`.
- For large files, free GPU memory between runs: `del model; torch.cuda.empty_cache()`.
