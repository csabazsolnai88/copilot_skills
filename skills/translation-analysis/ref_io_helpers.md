# I/O Helpers — Reading & Writing Translation Data

Reusable functions for reading translation data in various formats and writing results.

---

## Reading a tabular file

```python
import pandas as pd

def read_tabular(path: str) -> pd.DataFrame:
    """Read a CSV, XLSX, or Parquet file into a DataFrame."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported tabular format: {path}")
```

## Reading three separate files

```python
import json
import logging

logger = logging.getLogger(__name__)

def read_jsonl(path: str, text_key: str = "text") -> list[str]:
    """Read a JSONL file and extract text using text_key, with fallbacks.

    Skips blank lines. Falls back to common key names if text_key is absent.
    Logs a warning and appends an empty string for unparseable lines.
    """
    FALLBACK_KEYS = ("text", "segment", "src", "tgt", "translation")
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                texts.append("")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON on %s:%d — skipping", path, lineno)
                texts.append("")
                continue
            if text_key in obj:
                texts.append(obj[text_key])
            else:
                for k in FALLBACK_KEYS:
                    if k in obj:
                        texts.append(obj[k])
                        break
                else:
                    logger.warning("Key %r not found on %s:%d — skipping", text_key, path, lineno)
                    texts.append("")
    return texts

def read_moses(path: str) -> list[str]:
    """Read a plain-text Moses file (one segment per line)."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]
```

## Writing results

```python
import pandas as pd

def write_tabular(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, XLSX, or Parquet based on the file extension."""
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".xlsx"):
        df.to_excel(path, index=False)
    elif path.endswith(".parquet") or path.endswith(".pq"):
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
```

## Polars variants (for LLM-as-judge mode)

```python
import polars as pl
from pathlib import Path

def load_input(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        return pl.read_excel(path)
    else:
        raise ValueError(f"Unsupported: {suffix}")

def save_output(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix in (".parquet", ".pq"):
        df.write_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported: {suffix}")
```
