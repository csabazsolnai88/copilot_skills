---
name: vllm-app
description: Create vLLM client applications that process tabular data, send prompts to a running vLLM server, collect structured JSON responses, and write enriched tables. Always generate a launch_vllm_server.sh script for server startup.
---
## vLLM Server Launch Script

Every vllm app must include a short bash script for launching the vLLM server. Place this script in the working directory as `launch_vllm_server.sh` unless the user specifies another location or model.

**Example launch script:**

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

**Documentation:**
Add a note in the app’s README or docstring referencing the launch script and its purpose, e.g.:

> "To launch the vLLM server for this app, run `bash launch_vllm_server.sh` in your working directory. Adjust CUDA_VISIBLE_DEVICES and model name as needed."
# vLLM App Skill

Create applications that send tabular data to a running vLLM server, collect structured JSON responses, and write the enriched data back to disk.

## Assumptions

- A vLLM server is **already running** and accessible via an OpenAI-compatible HTTP API.
- Input is **tabular data** (CSV, Parquet, XLSX, or similar) loaded with Polars unless specified otherwise.
- Output is also tabular — the original table enriched with new columns derived from LLM responses.
- The reusable `VLLMClient` class (see `vllm_client.py` in this skill directory) handles all HTTP, retry, concurrency, and JSON-parsing concerns. **Do not rewrite the client; import and use it as-is.**

## Step-by-step workflow

Every vLLM app follows these steps in order. The agent MUST produce code for each step.

**File location rule:**
By default, generated app scripts (e.g., translation_review_app.py) should be saved in the current working directory unless the user requests integration into an existing project (such as adding the entrypoint to src/termforce/ or another project folder).

**vllm_client.py location rule:**
Always copy vllm_client.py to the same directory as the app entrypoint unless the user requests project integration. The app should import VLLMClient and AppSettings from vllm_client.py in the same directory. If the user requests integration, place vllm_client.py and update imports as needed.

### Step 1 — Define the task and identify columns

Read the user's description carefully. Determine:

1. **Input columns** — which columns from the table are injected into the prompt.
2. **Output columns** — which new columns the LLM response will produce.
3. **Row grouping** — whether rows are processed individually or in batches (default: batches of `batch_size` rows per prompt).

### Step 2 — Design the prompt

Write a `build_prompt(rows: list[dict], ...) -> str` function.

Rules:
- Present input rows as a **numbered list** so the LLM can reference them by index.
- Include a short **role/context** sentence (e.g., "You are a translation quality reviewer.").
- Include a **"Do NOT"** list of common mistakes to avoid.
- End with an explicit instruction to return **only JSON** matching the schema.
- Include one **concrete example** of input → expected JSON output inside the prompt.

Template:

```python
def build_prompt(rows: list[dict]) -> str:
    """Build prompt for <TASK DESCRIPTION>.

    Args:
        rows: List of row dicts from the input table.

    Returns:
        Formatted prompt string.
    """
    prompt = (
        "You are a <ROLE>. <TASK CONTEXT>.\n\n"
        "<INSTRUCTIONS>\n\n"
        "Do NOT:\n"
        "- <COMMON MISTAKE 1>\n"
        "- <COMMON MISTAKE 2>\n\n"
        "Example:\n"
        "Input:\n"
        "1. <EXAMPLE ROW>\n\n"
        "Expected output:\n"
        '{"results": [{"row_id": 1, "<OUTPUT_KEY>": "<EXAMPLE VALUE>"}]}\n\n'
        "Now process the following rows:\n\n"
    )
    for i, row in enumerate(rows, start=1):
        prompt += f"{i}. <FORMAT ROW FIELDS>\n"

    prompt += (
        "\nReturn ONLY a JSON object matching the schema. "
        "No explanation, no markdown.\n"
    )
    return prompt
```

### Step 3 — Design the response JSON schema

Write a `get_response_schema() -> dict` function that returns a JSON Schema dict.

Rules:
- **Always use flat data structures.** The top-level object contains a single array key (e.g. `"results"`).
- Each element in the array is a flat record — **no nested lists or dicts inside a record**.
- Always include a `row_id` (integer) so responses can be joined back to input rows.
- Keep value types simple: `string`, `number`, `integer`, `boolean`.

Template:

```python
def get_response_schema() -> dict:
    """JSON schema for guided generation."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_id": {"type": "integer"},
                        # One key per output column:
                        "<output_col>": {"type": "string"},
                    },
                    "required": ["row_id", "<output_col>"],
                },
            }
        },
        "required": ["results"],
    }
```

### Step 4 — Implement batching and processing loop

Use the same batching pattern as the existing extraction pipeline.

```python
import asyncio
import logging
from pathlib import Path

import polars as pl
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


def load_input(path: Path) -> pl.DataFrame:
    """Load tabular input. Supports csv, parquet, xlsx."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    elif suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        return pl.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_output(df: pl.DataFrame, path: Path) -> None:
    """Save tabular output. Supports csv, parquet, xlsx."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix == ".parquet":
        df.write_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def create_batches(df: pl.DataFrame, batch_size: int) -> list[list[dict]]:
    """Split dataframe rows into batches of dicts."""
    rows = df.to_dicts()
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


async def process_table(
    df: pl.DataFrame,
    client,            # VLLMClient instance
    batch_size: int,
    concurrency: int,
    build_prompt_fn,   # callable(rows) -> str
    schema: dict,
    output_key: str = "results",
) -> list[dict]:
    """Send all rows through the LLM in batches and collect results.

    Args:
        df: Input dataframe (must have a row_id column or one will be added).
        client: An initialized VLLMClient (inside async with).
        batch_size: Number of rows per prompt.
        concurrency: How many prompts to send in parallel.
        build_prompt_fn: Function that takes list[dict] and returns prompt str.
        schema: JSON schema dict for guided generation.
        output_key: Top-level key in the LLM response that contains the array.

    Returns:
        Flat list of result dicts (one per input row).
    """
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")

    batches = create_batches(df, batch_size)
    all_results = []

    for i in tqdm(range(0, len(batches), concurrency), desc="Processing"):
        batch_group = batches[i : i + concurrency]
        prompts = [build_prompt_fn(batch) for batch in batch_group]
        responses = await client.batch_complete(prompts, response_schema=schema)

        for batch, response in zip(batch_group, responses, strict=False):
            if isinstance(response, Exception):
                logger.error(f"Batch failed: {response}")
                continue
            try:
                results = response.get(output_key, [])
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")

    return all_results
```

### Step 5 — Assemble the CLI / main function

Wire everything together: load → process → join → save.

```python
async def run_app(
    input_path: Path,
    output_path: Path,
    vllm_url: str = "http://localhost:8000",
    batch_size: int = 5,
    concurrency: int = 8,
) -> None:
    """Run the vLLM app end-to-end."""
    from vllm_client import VLLMClient, AppSettings

    config = AppSettings(
        vllm_api_url=vllm_url,
        batch_size=batch_size,
        vllm_concurrency=concurrency,
    )

    df = load_input(input_path)

    # Add row_id for joining results back
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")

    schema = get_response_schema()

    async with VLLMClient(config) as client:
        results = await process_table(
            df, client, batch_size, concurrency, build_prompt, schema
        )

    # Join results back to original dataframe
    if results:
        results_df = pl.DataFrame(results)
        df = df.join(results_df, on="row_id", how="left")
    else:
        logger.warning("No results returned from LLM")

    # Drop row_id helper column if not in original data
    if "row_id" in df.columns:
        df = df.drop("row_id")

    save_output(df, output_path)
    logger.info(f"Saved enriched table to {output_path}")


if __name__ == "__main__":
    import sys
    asyncio.run(run_app(
        input_path=Path(sys.argv[1]),
        output_path=Path(sys.argv[2]),
    ))
```

---

## Complete worked example

**User request:** "I have a CSV with columns `src_segment`, `src_language`, `trg_segment`, `trg_language`, `translation`. `trg_segment` is a reference translation and `translation` is a machine translation that might have problems. Create a vllm-app that adds a column `issues` describing translation problems. Save as XLSX."

### Prompt function

```python
def build_prompt(rows: list[dict]) -> str:
    """Build prompt for translation quality review."""
    prompt = (
        "You are a professional translation quality reviewer. "
        "For each numbered entry below, compare the machine translation against "
        "the reference translation and identify issues in the machine translation.\n\n"
        "Issue categories: mistranslation, omission, addition, grammar, "
        "terminology, style, punctuation.\n\n"
        "Do NOT:\n"
        "- Invent issues that do not exist\n"
        "- Comment on the reference translation\n"
        "- Return anything other than JSON\n\n"
        "Example:\n"
        "Input:\n"
        '1. SRC: "The policyholder must pay the premium." '
        'REF: "Der Versicherungsnehmer muss die Prämie zahlen." '
        'MT: "Der Versicherer muss die Prämie zahlen."\n\n'
        "Expected output:\n"
        '{"results": [{"row_id": 1, "issues": '
        '"mistranslation: policyholder translated as insurer instead of '
        'policyholder"}]}\n\n'
        "Now review the following entries:\n\n"
    )
    for row in rows:
        rid = row["row_id"]
        src = row["src_segment"]
        ref = row["trg_segment"]
        mt = row["translation"]
        prompt += (
            f'{rid}. SRC ({row["src_language"]}): "{src}" '
            f'REF ({row["trg_language"]}): "{ref}" '
            f'MT: "{mt}"\n'
        )
    prompt += (
        "\nReturn ONLY a JSON object with a 'results' array. "
        "Each element must have 'row_id' (integer) and 'issues' (string). "
        "If no issues, set issues to 'none'. No markdown, no explanation.\n"
    )
    return prompt
```

### Response schema

```python
def get_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_id": {"type": "integer"},
                        "issues": {"type": "string"},
                    },
                    "required": ["row_id", "issues"],
                },
            }
        },
        "required": ["results"],
    }
```

### Full script (`translation_review_app.py`)

```python
"""Translation quality review vLLM app."""

import asyncio
import logging
from pathlib import Path

import polars as pl
from tqdm.asyncio import tqdm

# Import the reusable client from the skill directory
from vllm_client import VLLMClient, AppSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_prompt(rows: list[dict]) -> str:
    """Build prompt for translation quality review."""
    prompt = (
        "You are a professional translation quality reviewer. "
        "For each numbered entry below, compare the machine translation against "
        "the reference translation and identify issues in the machine translation.\n\n"
        "Issue categories: mistranslation, omission, addition, grammar, "
        "terminology, style, punctuation.\n\n"
        "Do NOT:\n"
        "- Invent issues that do not exist\n"
        "- Comment on the reference translation\n"
        "- Return anything other than JSON\n\n"
        "Example:\n"
        "Input:\n"
        '1. SRC: "The policyholder must pay the premium." '
        'REF: "Der Versicherungsnehmer muss die Prämie zahlen." '
        'MT: "Der Versicherer muss die Prämie zahlen."\n\n'
        "Expected output:\n"
        '{"results": [{"row_id": 1, "issues": '
        '"mistranslation: policyholder translated as insurer instead of '
        'policyholder"}]}\n\n'
        "Now review the following entries:\n\n"
    )
    for row in rows:
        rid = row["row_id"]
        src = row["src_segment"]
        ref = row["trg_segment"]
        mt = row["translation"]
        prompt += (
            f'{rid}. SRC ({row["src_language"]}): "{src}" '
            f'REF ({row["trg_language"]}): "{ref}" '
            f'MT: "{mt}"\n'
        )
    prompt += (
        "\nReturn ONLY a JSON object with a 'results' array. "
        "Each element must have 'row_id' (integer) and 'issues' (string). "
        "If no issues, set issues to 'none'. No markdown, no explanation.\n"
    )
    return prompt


def get_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_id": {"type": "integer"},
                        "issues": {"type": "string"},
                    },
                    "required": ["row_id", "issues"],
                },
            }
        },
        "required": ["results"],
    }


def load_input(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    elif suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        return pl.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_output(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix == ".parquet":
        df.write_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def create_batches(df: pl.DataFrame, batch_size: int) -> list[list[dict]]:
    rows = df.to_dicts()
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


async def process_table(df, client, batch_size, concurrency, build_prompt_fn, schema):
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")
    batches = create_batches(df, batch_size)
    all_results = []
    for i in tqdm(range(0, len(batches), concurrency), desc="Processing"):
        batch_group = batches[i : i + concurrency]
        prompts = [build_prompt_fn(batch) for batch in batch_group]
        responses = await client.batch_complete(prompts, response_schema=schema)
        for batch, response in zip(batch_group, responses, strict=False):
            if isinstance(response, Exception):
                logger.error(f"Batch failed: {response}")
                continue
            try:
                all_results.extend(response.get("results", []))
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
    return all_results


async def run_app(input_path: Path, output_path: Path, vllm_url="http://localhost:8000",
                  batch_size=5, concurrency=8):
    config = AppSettings(vllm_api_url=vllm_url, batch_size=batch_size,
                         vllm_concurrency=concurrency)
    df = load_input(input_path)
    if "row_id" not in df.columns:
        df = df.with_row_index("row_id")
    schema = get_response_schema()
    async with VLLMClient(config) as client:
        results = await process_table(df, client, batch_size, concurrency,
                                      build_prompt, schema)
    if results:
        results_df = pl.DataFrame(results)
        df = df.join(results_df, on="row_id", how="left")
    else:
        logger.warning("No results returned from LLM")
    if "row_id" in df.columns:
        df = df.drop("row_id")
    save_output(df, output_path)
    logger.info(f"Saved enriched table to {output_path}")


if __name__ == "__main__":
    import sys
    asyncio.run(run_app(Path(sys.argv[1]), Path(sys.argv[2])))
```

---

## Schema design guidelines

1. **Flat records only.** Every record in the results array must have simple scalar values — no nested lists, no nested dicts.
2. **Always include `row_id`** (integer) to join responses back to input rows.
3. **Use `string` for free-text outputs** (e.g. issues, summaries, translations).
4. **Use `number` for scores** — add `minimum` / `maximum` constraints when appropriate.
5. **Use `boolean` for binary flags.**

Bad (nested):
```json
{"row_id": 1, "issues": [{"type": "grammar", "detail": "wrong case"}]}
```

Good (flat):
```json
{"row_id": 1, "issues": "grammar: wrong case"}
```

If you need structured issue types, use separate columns:
```json
{"row_id": 1, "issue_type": "grammar", "issue_detail": "wrong case"}
```

## Prompt design checklist

- [ ] Role sentence ("You are a ...")
- [ ] Task context (domain, what the data represents)
- [ ] Clear instructions on what to produce
- [ ] "Do NOT" list (at least 2 items)
- [ ] One concrete example (input + expected JSON)
- [ ] Numbered input rows with relevant columns
- [ ] Explicit "Return only JSON" instruction at the end
- [ ] Mention the exact schema keys the response must contain

## VLLMClient reference

The reusable client is in `vllm_client.py` in this skill directory. Key API:

```python
from vllm_client import VLLMClient, AppSettings

config = AppSettings(
    vllm_api_url="http://localhost:8000",  # TERMFORCE_VLLM_API_URL
    vllm_api_key=None,                     # TERMFORCE_VLLM_API_KEY
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

Author: Agent Skill Generator
Date: 2026-02-23
