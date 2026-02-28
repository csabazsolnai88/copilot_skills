# MQM Prompt Library — Terminology / Inconsistent with Terminology Resource

This file contains prompt templates and checker code for the **MQM Terminology / Inconsistent with terminology resource** sub-type.
It is referenced from [`SKILL.md`](SKILL.md) in the *MQM Prompt Library* section.

**Two approaches are documented here:**

| Approach | When to use | File |
|----------|-------------|------|
| **1. Termbase-driven (preferred)** | A termbase CSV with `src_term`/`trg_term` columns is available | This file — see [§ Approach 1](#approach-1-termbase-driven-preferred) |
| **2. LLM-based (fallback)** | No termbase available; rely on LLM + reference translation | This file — see [§ Approach 2](#approach-2-llm-based-fallback) |

The MQM columns **must not be written** if neither a termbase nor an LLM is configured.

---

## Background: MQM Terminology / Inconsistent with Terminology Resource

Source: https://themqm.org/the-mqm-full-typology/ (CC BY 4.0)

**Category tree:** Terminology → Inconsistent with terminology resource

**Short definition:**
> A term is used inconsistently with a specified terminology resource (e.g. a
> project glossary, a subject-field standard, or organizational/product terminology).
> The target content does not use the term specified by the terminology resource for
> the corresponding source concept.

**Key criteria — flag ONLY when ALL of the following hold:**

1. The source word or phrase is a **domain-specific or organization-specific term** (not a
   common general-language word such as a copula, common verb, or generic preposition).
2. A **normative target equivalent** exists — it appears in the reference translation, in
   a known glossary, or is established by the subject field (e.g. regulatory/financial
   terminology).
3. The MT uses a **different word or phrase** that is not an accepted synonym for that term
   in the domain.

**Do NOT flag as wrong term:**
- Stylistic paraphrases of common verbs (e.g. "is not allowed" vs "may not" for "darf nicht").
- Near-synonyms for generic words ("clients" vs "customers" for "Kunden" without domain
  evidence that one is normative).
- Word-order differences or minor grammatical variation that do not change the term itself.
- Segments where the MT translation happens to differ from the reference but is still a
  plausible domain translation.
- Cases where the reference itself appears to use an unusual or less accurate term and the
  MT uses a more standard one — do not flag just because MT ≠ reference.
- Abbreviations or system codes that both the reference and the MT leave untranslated
  (e.g. "VV-Mandate", "AUX", "ACS", "BP", "IPS"). These are intentional.
- Compound German nouns that do not map to a single, well-known English technical term
  in the banking domain (e.g. "Teilvermögen" → "partial asset" vs "sub-asset" are both
  reasonable; only flag if the normative term is distinctly established in a glossary or
  regulatory standard — not just because the reference chose a different rendering).

---

## Output Format

Instead of a simple True/False flag the prompt extracts **the list of source-language terms
that were wrongly translated in the MT**. An empty list means no wrong terms detected.

```
JSON schema per row:
{
  "row_id": <int>,
  "wrong_terms": ["<source_lang_term_1>", ...]   // empty list if no errors
}
```

Downstream code can derive a boolean column as `len(wrong_terms) > 0` and also inspect
which specific terms triggered the flag.

---

## Annotated Examples from the Banking/Financial Domain

The following examples are drawn from a German→English banking translation memory.
They demonstrate the calibration the model should apply.

### Positive examples — wrong term IS present

| # | Source (DE) | Reference (EN) | MT (EN) | Wrong terms |
|---|-------------|----------------|---------|-------------|
| A | `#124 - Kapitalausstand Transfer nach Kapitalbuchungen` | `#124 - Outstanding capital transfer after capital postings` | `#124 - Capital Outstanding Transfer to Capital Accounting` | `Kapitalbuchungen` (→ "Capital Accounting"; must be "capital postings" — established accounting term) |
| B | `!Limite hat mehrere Verzugszinskonti Migration` | `!Limit has several default interest accounts, migrated` | `!Limite has several overdue interest accounts Migration` | `Verzugszinskonti` (→ "overdue interest accounts"; must be "default interest accounts" — established banking term for penalty interest); `Limite` (left untranslated — in the reference the German system term is rendered as "Limit") |
| C | `Teilvermögen darf nicht vom Typ "Vermögensverwaltung" sein` | `Container type must not be "Discretionary Portfolio"` | `Partial assets may not be of the "asset management" type` | `Vermögensverwaltung` (→ "asset management"; must be "Discretionary Portfolio" — established financial portfolio-management term) |
| D | `&1Die Angemessenheitsprüfung konnte nicht durchgeführt werden, weil der Vermögenswert nicht zu einer Kenntnis- und Erfahrungskategorie gehört.&2` | `&1The appropriateness test could not be evaluated because the asset does not belong to a knowledge and experience category.&2` | `&1The suitability test could not be carried out because the asset does not belong to a knowledge and experience category.&2` | `Angemessenheitsprüfung` (→ "suitability test"; must be "appropriateness test" — these are legally distinct MiFID II concepts: "Angemessenheitsprüfung" is the appropriateness check for non-advised services; "Geeignetheitsprüfung" is the suitability check for advice) |
| E | `Danach kann das Mandat auf «In Saldierung» geschlüsselt werden.` | `After that the IPS can be set to state 'Closing'.` | `After that, the mandate can be encrypted to "In Saldierung".` | `geschlüsselt` (→ "encrypted"; in this context means "set to status" — "encrypted" is a false cognate deriving from "Schlüssel" = key/code but the phrase means to assign a coded status, not to encrypt data); `In Saldierung` (left untranslated — reference renders this as the state label 'Closing') |

### Negative examples — wrong term is NOT present

| # | Source (DE) | Reference (EN) | MT (EN) | Reason no flag |
|---|-------------|----------------|---------|----------------|
| F | `!AUX Zinskonto hat andere Währung als AUX Kapitalkonto` | `!AUX interest account has another currency than AUX capital account` | `!AUX interest account has a different currency than AUX capital account` | "another" vs "a different" — purely stylistic synonyms; no domain term involved |
| G | `Die weiteren VV-Mandate bilden diesen steuerlichen Zustand nicht ab und können daher nicht geführt werden.` | `The other VV mandates do not reflect this tax situation and therefore cannot be managed.` | `The other VV mandates do not reflect this tax situation and therefore cannot be carried out.` | "managed" vs "carried out" for "geführt" — general-language verb; neither is a domain-specific corporate term here |
| H | `Dieses Mandat ist für Kunden mit Domizil Schweiz nicht empfohlen.` | `This mandate is not recommended for clients domiciled in Switzerland.` | `This mandate is not recommended for customers domiciled in Switzerland.` | "clients" vs "customers" for "Kunden" — both are acceptable general English translations; no evidence that one is normative in this context |
| I | `Das Konto ist für diesen Zweck nicht mehr zugelassen.` | `The account is no longer authorised for this purpose.` | `The account is no longer authorised for this purpose.` | Identical — no errors |
| J | `Es bieten sich alternative Anlageformen in unserer Produktepalette an.` | `Alternative investment forms are available in our product range.` | `Alternative investment options are available in our product range.` | "investment forms" vs "investment options" — neither is an established normative term; both are general paraphrases |

---

## Approach 1: Termbase-driven (preferred)

### Algorithm

1. Load the termbase CSV (columns: `src_term`, `trg_term`).
2. Pre-compute per-entry:
   - Normalised source string (`lower`, strip quotes, collapse whitespace).
   - Normalised target string (same + collapse hyphens to spaces for flexible matching).
   - spaCy lemma sequence of the source term (handles inflection/compounding).
   - Compiled word-boundary regex `(?<!\w)TERM(?!\w)` on the normalised source.
3. For each `(src, mt)` pair, scan all termbase entries:
   - **Pass 1** — word-boundary regex match on normalised `src` text.
   - **Pass 2** — spaCy lemma sliding-window on `src` lemma sequence (catches inflected forms not matched by the regex).
4. A match is a **violation** only when the expected target translation is **absent** from the (hyphen-normalised) MT.
5. Deduplicate violations:
   - Group by normalised expected target → keep the longest matching source term.
   - Then remove any source term that is a sub-string of a longer kept entry.

### Code

```python
import re
import spacy
import pandas as pd
from tqdm import tqdm

_QUOTE_RE = re.compile(r"['''\"]")
_WS_RE = re.compile(r"\s+")

def _normalize(s: str) -> str:
    s = _QUOTE_RE.sub("", s.lower().strip())
    return _WS_RE.sub(" ", s).strip()

def _normalize_trg(s: str) -> str:
    return _normalize(s).replace("-", " ")

def _lemma_seq(nlp, text: str) -> tuple[str, ...]:
    return tuple(
        t.lemma_.lower() for t in nlp(text)
        if not t.is_punct and not t.is_space and t.text.strip()
    )

def _src_pattern(term_norm: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)")


def load_termbase(path: str) -> tuple[list[dict], object]:
    """Load termbase CSV and pre-compute all matching helpers."""
    nlp = spacy.load("de_core_news_sm")  # adjust language model as needed
    tb = pd.read_csv(path)
    records = []
    for row in tqdm(tb.itertuples(index=False), total=len(tb), desc="indexing termbase"):
        src_norm = _normalize(row.src_term)
        records.append({
            "src_term":    row.src_term,
            "trg_term":    row.trg_term,
            "src_norm":    src_norm,
            "trg_norm":    _normalize_trg(row.trg_term),
            "src_lemmas":  _lemma_seq(nlp, row.src_term),
            "src_pattern": _src_pattern(src_norm),
            "src_len":     len(src_norm),
        })
    return records, nlp


def check_segment(src: str, mt: str, records: list[dict], nlp) -> list[dict]:
    """Return deduplicated list of termbase violations for one (src, mt) pair.

    Each violation dict has keys: src_term, exp_trg, match_type ("substr"|"lemma").
    Returns [] when no violations are found.
    """
    src_norm  = _normalize(src)
    mt_norm   = _normalize_trg(mt)
    src_lems  = _lemma_seq(nlp, src)
    n         = len(src_lems)
    raw: list[dict] = []

    for row in records:
        k       = len(row["src_lemmas"])
        matched = False
        mtype   = None

        if row["src_pattern"].search(src_norm):         # Pass 1
            matched, mtype = True, "substr"
        elif 1 <= k <= n:                                # Pass 2
            for i in range(n - k + 1):
                if src_lems[i:i+k] == row["src_lemmas"]:
                    matched, mtype = True, "lemma"
                    break

        if not matched:
            continue
        if row["trg_norm"] and row["trg_norm"] in mt_norm:
            continue                                     # correct translation present

        raw.append({"src_term": row["src_term"], "src_norm": row["src_norm"],
                    "exp_trg": row["trg_term"], "match_type": mtype,
                    "src_len": row["src_len"]})

    if not raw:
        return []

    # Dedup: keep longest src per unique target; then remove sub-string sources
    by_trg: dict[str, dict] = {}
    for v in raw:
        key = _normalize(v["exp_trg"])
        if key not in by_trg or v["src_len"] > by_trg[key]["src_len"]:
            by_trg[key] = v

    survivors = sorted(by_trg.values(), key=lambda v: v["src_len"], reverse=True)
    final, covered = [], set()
    for v in survivors:
        if any(v["src_norm"] in kept for kept in covered):
            continue
        final.append(v)
        covered.add(v["src_norm"])

    return [{"src_term": v["src_term"], "exp_trg": v["exp_trg"],
             "match_type": v["match_type"]} for v in final]
```

### Integration Snippet

```python
records, nlp = load_termbase("output/heuristicv1_6/ramt/de_en/terminology_suitable.csv")

violations = [check_segment(src, mt, records, nlp) for src, mt in zip(srcs, mts)]

df["mqm_wrong_terms"]           = [json.dumps(v, ensure_ascii=False) for v in violations]
df["mqm_terminology_wrong_term"] = [len(v) > 0 for v in violations]
```

**Notes:**
- The termbase may contain auto-generated terms with imperfect translations — that is expected and handled gracefully.
- Requires `spacy` and the appropriate language model (e.g. `python -m spacy download de_core_news_sm`).
- The MQM columns **must be omitted** if no termbase is provided; do not fall back to flagging all segments.

---

## Approach 2: LLM-based (fallback)

```python
_SYSTEM_PROMPT = """\
You are an expert translation reviewer specializing in the MQM (Multidimensional Quality
Metrics) typology for banking and financial texts.
"""

_TASK_PROMPT_HEADER = """\
Task: Identify MQM Terminology / Wrong term errors in machine translations.

Definition (MQM Terminology / Wrong term):
  A term in the MT is NOT the correct, normative equivalent of the corresponding
  source term, where a normative equivalent exists (from the reference, a glossary,
  or the subject field).

Flag a term as wrong ONLY when:
  1. It is a domain-specific or organizational term (NOT a common verb, preposition,
     or generic noun).
  2. A normative English equivalent is established (visible in the reference or
     in standard banking / regulatory usage).
  3. The MT uses a different word that is not an accepted domain synonym.

Do NOT flag:
  - Stylistic paraphrases of common verbs ("may not" vs "is not allowed").
  - "clients" vs "customers" unless there is domain evidence one is normative.
  - Minor word-order or grammatical variation.
  - Terms left in the source language ONLY if they are proper nouns / codes that
    should not be translated (e.g. "Avaloq-Z", "ACS", "FATCA", "AIA").

For each entry return the LIST of source-language terms that were wrongly translated.
Return an empty list when no wrong term is present.

Example (correct output for entry 1):
  Input:
    1. SRC: "#124 - Kapitalausstand Transfer nach Kapitalbuchungen"
       REF: "#124 - Outstanding capital transfer after capital postings"
       MT:  "#124 - Capital Outstanding Transfer to Capital Accounting"
  Output entry: {"row_id": 1, "wrong_terms": ["Kapitalbuchungen"]}

Example (correct output for entry 2):
  Input:
    2. SRC: "!AUX Zinskonto hat andere Währung als AUX Kapitalkonto"
       REF: "!AUX interest account has another currency than AUX capital account"
       MT:  "!AUX interest account has a different currency than AUX capital account"
  Output entry: {"row_id": 2, "wrong_terms": []}

Entries to evaluate:
"""

_TASK_PROMPT_FOOTER = """\

Return ONLY a single JSON object — no markdown, no commentary.
Schema:
{
  "results": [
    {"row_id": <int>, "wrong_terms": [<source_lang_str>, ...]},
    ...
  ]
}
"""


def build_mqm_wrong_term_prompt(rows: list[dict]) -> str:
    """Build a Terminology/Wrong-term MQM prompt for a batch of translation triples.

    Args:
        rows: List of dicts with keys row_id, src, trg (reference), mt.

    Returns:
        Prompt string ready to send to the LLM.
    """
    body = ""
    for r in rows:
        body += f'{r["row_id"]}. SRC: "{r["src"]}"\n'
        body += f'   REF: "{r["trg"]}"\n'
        body += f'   MT:  "{r["mt"]}"\n\n'
    return _SYSTEM_PROMPT + "\n" + _TASK_PROMPT_HEADER + body + _TASK_PROMPT_FOOTER
```

---

## Response Schema

```python
def get_mqm_wrong_term_schema() -> dict:
    """JSON schema for guided generation of MQM wrong-term results."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_id": {"type": "integer"},
                        "wrong_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source-language terms that were wrongly translated in the MT. Empty list if none.",
                        },
                    },
                    "required": ["row_id", "wrong_terms"],
                },
            }
        },
        "required": ["results"],
    }
```

---

## Integration Snippet

Drop into any script that already imports `VLLMClient`:

```python
from prompts_mqm import build_mqm_wrong_term_prompt, get_mqm_wrong_term_schema
import json

schema = get_mqm_wrong_term_schema()

# Prepare rows (must have: row_id, src, trg, mt)
rows = df.to_dicts()  # polars, or df.to_dict("records") for pandas

batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
prompts = [build_mqm_wrong_term_prompt(b) for b in batches]

responses = await client.batch_complete(prompts, response_schema=schema)

# Flatten and join
all_results = []
for resp in responses:
    if isinstance(resp, Exception):
        continue
    all_results.extend(resp.get("results", []))

# Map back to DataFrame — store as JSON string for CSV compatibility
result_map = {r["row_id"]: r["wrong_terms"] for r in all_results}
df["mqm_wrong_terms"] = [
    json.dumps(result_map.get(rid, []), ensure_ascii=False)
    for rid in df["row_id"]
]
df["mqm_terminology_wrong_term"] = df["mqm_wrong_terms"].map(
    lambda s: len(json.loads(s)) > 0
)
```
