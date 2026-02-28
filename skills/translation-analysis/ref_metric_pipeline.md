# Metric Pipeline — BLEU, chrF, Structural Checks & Aggregation

Automatic pairwise metrics for translation triples. No LLM required.

---

## Computing BLEU scores

```python
import sacrebleu

def bleu_score(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU for a single reference/hypothesis pair."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score
```

## Metric registry (extensible)

```python
from collections.abc import Callable

# Metric functions must have signature (reference: str, hypothesis: str) -> float | bool.
# Boolean metrics (e.g. structural checks) are stored as 0/1 in the output table.
METRICS: dict[str, Callable[[str, str], float | bool]] = {
    "bleu": bleu_score,
    # "chrf": chrf_score,
    # "ter":  ter_score,
    # "overgeneration":  overgeneration_score,
    # "undergeneration": undergeneration_score,
}
```

## Full metric pipeline

```python
from __future__ import annotations

import json
from collections.abc import Callable
from multiprocessing import get_context

import pandas as pd
import sacrebleu
from tqdm import tqdm


def bleu_score(reference: str, hypothesis: str) -> float:
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


def overgeneration_score(reference: str, hypothesis: str) -> bool:
    """True when the MT contains more newlines than the reference."""
    return hypothesis.count("\n") > reference.count("\n")


def undergeneration_score(reference: str, hypothesis: str) -> bool:
    """True when the MT contains fewer newlines than the reference."""
    return hypothesis.count("\n") < reference.count("\n")


# Metric functions must have signature (reference, hypothesis) -> float | bool.
# Boolean metrics are stored as 0/1 in the output table.
METRICS: dict[str, Callable[[str, str], float | bool]] = {
    "bleu": bleu_score,
    "overgeneration": overgeneration_score,
    "undergeneration": undergeneration_score,
}


# Must be a top-level function so the spawn pool can pickle it.
def _score_row(args: tuple[str, str, str, list[str]]) -> dict:
    src, trg, mt, metric_names = args
    row: dict = {"src": src, "trg": trg, "mt": mt}
    for name in metric_names:
        row[name] = METRICS[name](trg, mt)
    return row


def run_metric_analysis(
    src_segments: list[str],
    trg_segments: list[str],
    mt_segments: list[str],
    output_path: str,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    if metric_names is None:
        metric_names = ["bleu"]
    args = [
        (s, t, m, metric_names)
        for s, t, m in zip(src_segments, trg_segments, mt_segments)
    ]
    # sacrebleu is single-threaded; use a spawned process pool to parallelise
    # across CPU cores. `spawn` (not `fork`) is required when the parent uses
    # any multithreaded library (e.g. Polars, numpy) to avoid mutex deadlocks.
    with get_context("spawn").Pool() as pool:
        records = list(
            tqdm(
                pool.imap(_score_row, args),
                total=len(args),
                desc="scoring",
                unit="seg",
            )
        )
    df = pd.DataFrame(records)
    write_tabular(df, output_path)
    return df
```

---

## Extending with New Metrics

Metric functions must have signature `(reference: str, hypothesis: str) -> float | bool`.
Boolean metrics are stored as `0`/`1` in the output table.
Register the function in the `METRICS` dict to make it available via `--metrics`.

### Example: chrF (float metric)

```python
def chrf_score(reference: str, hypothesis: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score

METRICS["chrf"] = chrf_score
```

### Example: structural newline checks (boolean metrics)

Overgeneration and undergeneration detect segments where the MT has a different
number of newlines than the reference — a common signal that the model generated
extra content or dropped output blocks.

```python
def overgeneration_score(reference: str, hypothesis: str) -> bool:
    """True when the MT contains more newlines than the reference."""
    return hypothesis.count("\n") > reference.count("\n")


def undergeneration_score(reference: str, hypothesis: str) -> bool:
    """True when the MT contains fewer newlines than the reference."""
    return hypothesis.count("\n") < reference.count("\n")


METRICS["overgeneration"] = overgeneration_score
METRICS["undergeneration"] = undergeneration_score
```

To run all metrics at once:

```python
df = run_metric_analysis(
    src_segments, trg_segments, mt_segments,
    output_path="results.csv",
    metric_names=["bleu", "overgeneration", "undergeneration"],
)
# overgeneration / undergeneration columns contain 0 or 1 (bool stored as int)
print(df[["bleu", "overgeneration", "undergeneration"]].describe())
```

---

## Aggregating segment-level scores to dataset-level metrics

When evaluating a model you often want a single score or small set of
statistics for the whole dataset (corpus BLEU, average neural metric, % of
overgeneration). The skill supports both segment-wise and corpus-level
aggregation. Examples below show common patterns.

1) Corpus BLEU (recommended when you want a corpus-level BLEU rather than
     the mean of sentence BLEUs):

```python
import sacrebleu

# hypotheses: list[str] (mt), references: list[str] (trg)
corpus = sacrebleu.corpus_bleu(hypotheses, [references])
print('Corpus BLEU:', corpus.score)
```

2) Average of a per-segment metric (e.g. MetricX or sentence-level BLEU):

```python
# df is the per-segment table with column 'metricx' or 'bleu'
mean_metric = df['metricx'].mean()
print('Mean metricX:', mean_metric)
```

3) Percentage for boolean structural checks (over/undergeneration):

```python
# columns are 0/1 booleans stored as integers
pct_over = df['overgeneration'].mean() * 100.0
pct_under = df['undergeneration'].mean() * 100.0
print(f"Overgeneration: {pct_over:.2f}%  Undergeneration: {pct_under:.2f}%")
```

Notes:
- Prefer `sacrebleu.corpus_bleu` or `BLEU.corpus_score` when reporting a
    single BLEU for the corpus — averaging sentence-level BLEU values is
    informative but not equivalent to corpus BLEU.
- For neural metrics (MetricX, COMET) report the mean (or median) and also
    a small set of robust statistics (std, 10/90 percentiles) for stability.
- When combining metrics into a single dashboard, normalize ranges (e.g.
    scale MetricX to 0-100 or swap as shown in the MetricX example) so values
    are directly comparable.
