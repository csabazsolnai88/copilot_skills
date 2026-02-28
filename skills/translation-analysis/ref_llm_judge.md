# LLM-as-Judge Mode — Prompts, Batching & App Skeleton

Use these patterns when the user asks for LLM-based critique, scoring, or error categorization of machine translations.

**Prerequisites:** Copy `vllm_client.py` from this skill directory to the same directory as the generated script.

---

## vLLM Server Launch Script

When generating an LLM-as-judge app, also generate a `launch_vllm_server.sh` script:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image":10}' \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2
```

Adjust model, GPU devices and parallelism to match the user's setup.

---

## VLLMClient Usage

Key API:

```python
from vllm_client import VLLMClient, AppSettings

config = AppSettings(
    vllm_api_url="http://localhost:8000",  # env: TERMFORCE_VLLM_API_URL
    vllm_api_key=None,                     # env: TERMFORCE_VLLM_API_KEY
    batch_size=5,                          # rows per prompt
    vllm_concurrency=8,                    # parallel requests
    request_timeout=120,                   # seconds
    max_retries=3,
    retry_backoff=1.0,
)

async with VLLMClient(config) as client:
    # Single request
    result = await client.complete(prompt, response_schema=schema)

    # Batch of prompts (concurrent)
    results = await client.batch_complete(prompts, response_schema=schema)
```

---

## Prompt Design Checklist

Every LLM judge prompt MUST include:
- [ ] Role sentence ("You are a …")
- [ ] Task context (domain, what the data represents)
- [ ] Clear instructions on what to produce
- [ ] "Do NOT" list (at least 2 items)
- [ ] One concrete example (input → expected JSON)
- [ ] Numbered input rows with `src`, `trg`, `mt` fields
- [ ] Explicit "Return only JSON" instruction at the end
- [ ] Mention the exact schema keys the response must contain

---

## Response Schema Design

Rules:
- **Always use flat data structures.** Top-level object contains a single array key `"results"`.
- Each element has `row_id` (integer) + output columns with scalar values.
- **No nested lists or dicts inside a record.**

```python
def get_response_schema() -> dict:
    """JSON schema for guided generation — adapt keys to the task."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_id": {"type": "integer"},
                        # Adapt these to the task:
                        "critique": {"type": "string"},
                        # "score": {"type": "number", "minimum": 0, "maximum": 100},
                        # "error_categories": {"type": "string"},
                    },
                    "required": ["row_id", "critique"],
                },
            }
        },
        "required": ["results"],
    }
```

---

## Batching and Processing Loop

```python
import asyncio
import logging
import polars as pl

logger = logging.getLogger(__name__)


def create_batches(df: pl.DataFrame, batch_size: int) -> list[list[dict]]:
    """Slice the DataFrame lazily so only one batch is materialised at a time."""
    return [df.slice(i, batch_size).to_dicts() for i in range(0, len(df), batch_size)]


async def process_table(
    df: pl.DataFrame,
    client,              # VLLMClient instance
    batch_size: int,
    concurrency: int,    # passed to AppSettings; the client semaphore enforces it
    build_prompt_fn,     # callable(rows) -> str
    schema: dict,
    output_key: str = "results",
) -> list[dict]:
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")
    batches = create_batches(df, batch_size)
    prompts = [build_prompt_fn(batch) for batch in batches]
    logger.info("Submitting %d batches to vLLM (max %d concurrent)", len(prompts), concurrency)
    # Submit everything at once; VLLMClient's internal semaphore caps live requests,
    # so slower batches do not block faster ones (no group-wait straggler problem).
    # tqdm.asyncio.tqdm.gather wraps asyncio.gather with a live progress bar.
    from tqdm.asyncio import tqdm as tqdm_asyncio
    responses = await tqdm_asyncio.gather(
        *[client.complete(p, response_schema=schema) for p in prompts],
        return_exceptions=True,
        desc="LLM batches",
        unit="batch",
    )
    all_results: list[dict] = []
    for response in responses:
        if isinstance(response, Exception):
            logger.error("Batch failed: %s", response)
            continue
        try:
            all_results.extend(response.get(output_key, []))
        except Exception as e:
            logger.error("Failed to parse response: %s", e)
    if len(all_results) != len(df):
        logger.warning(
            "Coverage mismatch: expected %d result rows, got %d",
            len(df),
            len(all_results),
        )
    return all_results
```

---

## Prompt Templates

### Translation Critique

```python
def build_critique_prompt(rows: list[dict]) -> str:
    """Build prompt asking LLM to critique machine translations."""
    prompt = (
        "You are a professional translation quality reviewer.\n"
        "For each numbered entry, compare the machine translation (MT) against "
        "the reference translation (REF) and write a brief critique of the MT.\n\n"
        "Focus on: accuracy, fluency, terminology, omissions, additions.\n\n"
        "Do NOT:\n"
        "- Invent issues that do not exist\n"
        "- Comment on the reference translation quality\n"
        "- Return anything other than JSON\n\n"
        "Example:\n"
        "Input:\n"
        '1. SRC: "The policyholder must pay the premium."\n'
        '   REF: "Le preneur d\'assurance doit payer la prime."\n'
        '   MT:  "L\'assureur doit payer la prime."\n\n'
        "Expected output:\n"
        '{"results": [{"row_id": 1, "critique": "Mistranslation: '
        "'policyholder' was incorrectly translated as 'assureur' (insurer) "
        "instead of 'preneur d'assurance'.\"}]}\n\n"
        "Now review the following entries:\n\n"
    )
    for row in rows:
        rid = row["row_id"]
        prompt += (
            f'{rid}. SRC: "{row["src"]}"\n'
            f'   REF: "{row["trg"]}"\n'
            f'   MT:  "{row["mt"]}"\n\n'
        )
    prompt += (
        "Return ONLY a JSON object with a 'results' array. "
        "Each element must have 'row_id' (integer) and 'critique' (string). "
        "If no issues, set critique to 'No issues found'. "
        "No markdown, no explanation.\n"
    )
    return prompt
```

### Error Categorization

```python
def build_error_categorization_prompt(rows: list[dict]) -> str:
    """Build prompt asking LLM to categorize MT errors."""
    prompt = (
        "You are a machine translation error analyst.\n"
        "For each entry, classify the errors in the machine translation.\n\n"
        "Error categories:\n"
        "- hallucination: MT contains information not in the source\n"
        "- deletion: MT omits information present in the source\n"
        "- mistranslation: a word or phrase is translated incorrectly\n"
        "- grammar: grammatical error in the MT\n"
        "- terminology: wrong domain-specific term used\n"
        "- style: unnatural phrasing (but meaning is correct)\n"
        "- none: no errors found\n\n"
        "Do NOT:\n"
        "- Assign categories that do not apply\n"
        "- Return anything other than JSON\n\n"
        "Example:\n"
        "Input:\n"
        '1. SRC: "The premium is due on the first of each month."\n'
        '   REF: "Die Prämie ist am Ersten jedes Monats fällig."\n'
        '   MT:  "Die Prämie ist fällig."\n\n'
        "Expected output:\n"
        '{"results": [{"row_id": 1, "error_categories": "deletion", '
        '"explanation": "MT omits the due date information."}]}\n\n'
        "Now analyze the following entries:\n\n"
    )
    for row in rows:
        rid = row["row_id"]
        prompt += (
            f'{rid}. SRC: "{row["src"]}"\n'
            f'   REF: "{row["trg"]}"\n'
            f'   MT:  "{row["mt"]}"\n\n'
        )
    prompt += (
        "Return ONLY a JSON object with a 'results' array. "
        "Each element must have 'row_id' (integer), "
        "'error_categories' (comma-separated string of categories), "
        "and 'explanation' (string). "
        "No markdown, no explanation outside the JSON.\n"
    )
    return prompt
```

### LLM Scoring (0–100)

```python
def build_scoring_prompt(rows: list[dict]) -> str:
    """Build prompt asking LLM to score MT quality 0-100."""
    prompt = (
        "You are a translation quality scorer.\n"
        "For each entry, assign a quality score from 0 to 100 for the MT.\n"
        "100 = perfect translation, 0 = completely wrong.\n\n"
        "Scoring guidelines:\n"
        "- 90-100: No errors, natural and accurate\n"
        "- 70-89: Minor issues (style, punctuation) but meaning preserved\n"
        "- 40-69: Some meaning errors or significant fluency issues\n"
        "- 0-39: Major errors, hallucinations, or mostly wrong\n\n"
        "Do NOT:\n"
        "- Give the same score to every entry\n"
        "- Return anything other than JSON\n\n"
        "Example:\n"
        "Input:\n"
        '1. SRC: "Please sign the contract."\n'
        '   REF: "Bitte unterschreiben Sie den Vertrag."\n'
        '   MT:  "Bitte unterschreiben Sie den Vertrag."\n\n'
        "Expected output:\n"
        '{"results": [{"row_id": 1, "score": 100, "justification": '
        '"Perfect translation."}]}\n\n'
        "Now score the following entries:\n\n"
    )
    for row in rows:
        rid = row["row_id"]
        prompt += (
            f'{rid}. SRC: "{row["src"]}"\n'
            f'   REF: "{row["trg"]}"\n'
            f'   MT:  "{row["mt"]}"\n\n'
        )
    prompt += (
        "Return ONLY a JSON object with a 'results' array. "
        "Each element must have 'row_id' (integer), 'score' (integer 0-100), "
        "and 'justification' (string). "
        "No markdown, no explanation.\n"
    )
    return prompt
```

---

## Extending with New LLM Judge Tasks

1. Write a new `build_*_prompt(rows)` function following the prompt checklist.
2. Write a matching `get_*_response_schema()`.

---

## Full LLM-as-Judge App Skeleton

```python
# %% Imports
import asyncio
import logging
from pathlib import Path

import polars as pl
from tqdm.asyncio import tqdm
from vllm_client import VLLMClient, AppSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% Prompt and schema (adapt to the specific task)

def build_prompt(rows: list[dict]) -> str:
    # Use one of the prompt templates above or design a custom one
    ...

def get_response_schema() -> dict:
    # Match the prompt's expected output
    ...

# %% I/O helpers
# (import from ref_io_helpers.md patterns)

# %% Processing

async def process_table(df, client, batch_size, concurrency, build_prompt_fn, schema):
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")
    batches = create_batches(df, batch_size)
    prompts = [build_prompt_fn(batch) for batch in batches]
    from tqdm.asyncio import tqdm as tqdm_asyncio
    responses = await tqdm_asyncio.gather(
        *[client.complete(p, response_schema=schema) for p in prompts],
        return_exceptions=True,
        desc="LLM batches",
        unit="batch",
    )
    all_results = []
    for response in responses:
        if isinstance(response, Exception):
            logger.error(f"Batch failed: {response}")
            continue
        try:
            all_results.extend(response.get("results", []))
        except Exception as e:
            logger.error(f"Failed to parse: {e}")
    if len(all_results) != len(df):
        logger.warning(f"Coverage mismatch: expected {len(df)} rows, got {len(all_results)}")
    return all_results

# %% Main

async def run_app(
    input_path: Path,
    output_path: Path,
    vllm_url: str = "http://localhost:8000",
    batch_size: int = 5,
    concurrency: int = 8,
) -> None:
    config = AppSettings(
        vllm_api_url=vllm_url,
        batch_size=batch_size,
        vllm_concurrency=concurrency,
    )
    df = load_input(input_path)
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")
    schema = get_response_schema()
    async with VLLMClient(config) as client:
        results = await process_table(
            df, client, batch_size, concurrency, build_prompt, schema
        )
    if results:
        results_df = pl.DataFrame(results)
        if len(results_df) != len(df):
            logger.warning(
                "Coverage mismatch after LLM: %d input rows but %d results — "
                "missing rows will have null LLM columns after the join",
                len(df), len(results_df),
            )
        df = df.join(results_df, on="row_id", how="left")
    else:
        logger.warning("No results returned from LLM")
    if "row_id" in df.columns:
        df = df.drop("row_id")
    save_output(df, output_path)
    logger.info(f"Saved {output_path}")

if __name__ == "__main__":
    import sys
    asyncio.run(run_app(Path(sys.argv[1]), Path(sys.argv[2])))
```
