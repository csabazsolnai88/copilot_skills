---
name: translation-analysis
description: Analyze machine translations using automatic metrics (BLEU, chrF, …) and/or LLM-as-a-judge evaluation via a vLLM server. Supports creating new scripts or extending existing ones. Always generates a notebook-style script (# %% cells).
---

# Copilot Skill: Translation Analysis

## Overview
This skill analyzes machine translations by:
- Computing **automatic pairwise metrics** (BLEU, chrF, TER, …) for translation triples (source, reference, machine translation).
- Using an **LLM as a judge** via a vLLM server to produce free-text critiques, error categories, or quality scores for machine translations.

It is designed to be **extensible** — new metrics and new LLM judge tasks can be added incrementally. The agent may either **create a new script from scratch** or **extend an existing script/application** depending on the user's prompt.

---

## When to Use

Use this skill when the user wants to:
- Compute BLEU / chrF / TER or other automatic translation quality metrics.
- Use an LLM to critique, score, or categorize errors in machine translations.
- Build or extend a script/notebook that processes translation triples.
- The user may refer to this as "translation-analysis", "translation analysis", "pairwise bleu", "llm judge", "translation review", "mt evaluation", or similar.

**How to decide the mode:**
| User mentions … | Mode |
|---|---|
| BLEU, chrF, TER, sacrebleu, automatic metric | **Metric mode** (no LLM needed) |
| LLM, judge, critique, review, error categories, hallucination, vLLM, score with LLM | **LLM-as-judge mode** (needs vLLM server) |
| Both | Combine both in the same script |

---

## Input Formats

The skill accepts **two input modes**. Determine which one to use based on the user's prompt.

### Mode A — Single Tabular File
A CSV, XLSX, or Parquet file where each row is a translation triple. The user will specify which columns contain the source, target, and machine translation segments.

### Mode B — Three Separate Files
Three files (one per segment type: source, target, machine translation), line-aligned. Supported formats:
- **JSONL**: each line is a JSON object; the user specifies the key that holds the text (e.g. `"text"`).
- **Moses / plain text**: one segment per line, plain UTF-8 text.

**Remote files:** If paths contain `:` (e.g. `host:/path/file.jsonl`), they are remote. The script should:
1. Check if the file already exists locally in `--tmp-dir` / basename.
2. If yes, use the local copy.
3. If no, `scp` it down.

---

## Output Format

The output is always a tabular file (CSV, XLSX, or Parquet — determined by the output path the user provides).

### Metric mode output
| src | trg | mt | bleu |
|-----|-----|----|------|

If additional metrics are computed, they appear as extra columns (e.g. `chrf`, `ter`).

### LLM-as-judge mode output
| src | trg | mt | *LLM output columns …* |
|-----|-----|----|------------------------|

The LLM output columns depend on the task: `critique`, `error_categories`, `score`, `issue_type`, etc.

---

## Creating vs. Extending

**Check the user's prompt carefully:**
- If they say "extend", "add to", "modify", or reference an existing script/file → **edit the existing file** to add the new capability.
- If they say "create", "make", "new", or give no existing file → **create a new script from scratch**.
- When extending, preserve all existing functionality and add the new feature alongside it.

---

## Steps Performed

### For metric mode:

1. **Parse the user prompt** to determine:
   - Input mode (tabular vs. three files).
   - File paths and column names / text keys.
   - Desired metric(s) (default: BLEU).
   - Output file path and format.

2. **Read inputs** into three aligned lists of strings: `src_segments`, `trg_segments`, `mt_segments`.

3. **Compute the requested metric(s)** for each `(trg, mt)` pair.

4. **Write the results** to the output file with columns: `src`, `trg`, `mt`, and one column per metric.

### For LLM-as-judge mode:

1. **Parse the user prompt** to determine:
   - Input mode (tabular vs. three files).
   - File paths and column names / text keys.
   - What the LLM should produce (critique, error categories, scores, etc.).
   - vLLM server URL (default: `http://localhost:8000`).
   - Output file path and format.

2. **Read inputs** into a table with at least `src`, `trg`, `mt` columns.

3. **Design the prompt** and **response schema** for the LLM task.

4. **Send translation triples to the vLLM server** in batches using `VLLMClient`.

5. **Join LLM responses** back to the original table.

6. **Write the enriched results** to the output file.

---

## Dependencies

### Metric mode
```bash
pip install sacrebleu pandas openpyxl
```

### LLM-as-judge mode
```bash
pip install aiohttp pydantic-settings polars tqdm
```

---

## Feature Reference Index

When implementing, read the dispatch info above first, then load **ONLY** the
reference files needed for the user's request. All files are in this skill directory.

| User wants …                              | Read these files                              |
|-------------------------------------------|-----------------------------------------------|
| Any new script (always)                   | `ref_io_helpers.md`                           |
| BLEU / chrF / TER / automatic metrics     | `ref_metric_pipeline.md`                      |
| MetricX (neural metric)                   | `ref_metricx.md`                              |
| LLM judge / critique / scoring / errors   | `ref_llm_judge.md`                            |
| MQM heuristic checks (no LLM)            | `ref_mqm_heuristics.md`                       |
| MQM terminology (termbase or LLM)         | `prompts_mqm.md`                              |
| Topic modeling (LDA / BERTopic)           | `ref_topic_modeling.md`                        |

**When in doubt, read the relevant reference file** — it contains full code samples,
prompt templates, and tuning guidance.

---

## VLLMClient (synopsis)

The reusable async client is in `vllm_client.py` in this skill directory. **Always copy it to the same directory as the generated app script and import from there.** Do not rewrite the client.

```python
from vllm_client import VLLMClient, AppSettings

config = AppSettings(
    vllm_api_url="http://localhost:8000",
    batch_size=5,
    vllm_concurrency=8,
)
async with VLLMClient(config) as client:
    result = await client.complete(prompt, response_schema=schema)
```

Full usage details are in `ref_llm_judge.md`.

---

## MQM Prompt Library

Reusable prompt templates and checker code for MQM error types:

| File | Covers |
|------|--------|
| `prompts_mqm.md` | **Terminology / Inconsistent with terminology resource** — (1) termbase-driven checker (preferred, deterministic, no LLM needed); (2) LLM-based fallback with annotated positive/negative examples |
| `ref_mqm_heuristics.md` | **Duplication, Entity, Hallucination, Addition, Omission** — five heuristic checks, no LLM required |

**How to use:**

1. **If a termbase is available** (a CSV with `src_term` / `trg_term` columns), use the `load_termbase` + `check_segment` functions from the *Approach 1* section of `prompts_mqm.md`. Pass `--termbase <path>` to the script. MQM columns are only written when this argument is provided.

2. **If no termbase is available**, fall back to the LLM prompt approach documented in *Approach 2* of `prompts_mqm.md`.

**Never write MQM columns when neither a termbase nor an LLM is configured** — an empty `mqm_wrong_terms` column is misleading.

---

## Example Prompts (routing guide)

These examples show how to map user requests to the correct mode and reference files.

| # | User prompt (summary) | Mode | Reference files to read |
|---|---|---|---|
| 1 | "3 JSONL files, compute BLEU" | Metric | `ref_io_helpers.md`, `ref_metric_pipeline.md` |
| 2 | "Single CSV, compute BLEU" | Metric | `ref_io_helpers.md`, `ref_metric_pipeline.md` |
| 3 | "LLM critique of translations" | LLM judge | `ref_io_helpers.md`, `ref_llm_judge.md` |
| 4 | "Extend existing script with LLM error categorization" | LLM judge (extend) | `ref_llm_judge.md` |
| 5 | "Score translations 0–100 with LLM" | LLM judge | `ref_io_helpers.md`, `ref_llm_judge.md` |
| 6 | "Run MQM heuristic checks" | Metric | `ref_mqm_heuristics.md` |
| 7 | "Topic modeling on segments" | Metric | `ref_topic_modeling.md` |
| 8 | "Full analysis: BLEU + MQM + topics" | Metric | `ref_io_helpers.md`, `ref_metric_pipeline.md`, `ref_mqm_heuristics.md`, `ref_topic_modeling.md` |
