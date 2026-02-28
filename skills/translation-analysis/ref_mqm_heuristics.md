# MQM Heuristic Quality Checks

Six MQM error types detected without an LLM — purely via heuristics, embeddings, and local NLP tools. Each check writes a boolean flag column (`mqm_{aspect}`) for counting plus a details/score column for diagnostics. All checks use the established skip-if-exists pattern.

| MQM path | Column prefix | Method | LLM required? |
|---|---|---|---|
| Linguistic conventions / Duplication | `mqm_duplication` | Regex: consecutive repeated words, repeated n-gram spans, repeated sentences | No |
| Accuracy / Mistranslation / Entity | `mqm_entity` | Regex + set comparison: numbers, identifiers, proper nouns | No |
| Accuracy / Mistranslation / MT hallucination | `mqm_hallucination` | Cross-lingual cosine similarity (LaBSE) between src ↔ mt | No (embeddings only) |
| Accuracy / Addition | `mqm_addition` | Length ratio mt/ref; flag when ratio > threshold | No |
| Accuracy / Omission | `mqm_omission` | Length ratio mt/ref; flag when ratio < threshold | No |
| Linguistic conventions / Grammar | `mqm_grammar` | LanguageTool local server — grammar rules (excluding spelling/style) | No (Java server) |

## Output columns per check

| Check | Flag column | Details column | Notes |
|---|---|---|---|
| Duplication | `mqm_duplication` (bool) | `mqm_duplication_details` (JSON list of strings) | Each string describes one duplication instance |
| Entity | `mqm_entity` (bool) | `mqm_entity_details` (JSON list of strings) | Describes missing/extra numbers, identifiers, proper nouns |
| Hallucination | `mqm_hallucination` (bool) | `mqm_hallucination_score` (float 0–1) | Cosine similarity; lower = more hallucinated |
| Addition | `mqm_addition` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Omission |
| Omission | `mqm_omission` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Addition |
| Grammar | `mqm_grammar` (bool) | `mqm_grammar_details` (JSON list of strings) + `mqm_grammar_count` (int) | Each string: `category/rule: "matched" → "fix" (message)` |

**Cross-flagging is expected.** A hallucinated segment may also be flagged for addition (different content = longer) and entity (entities don't match). A severely truncated segment may flag both omission and entity. When counting error frequencies, filter with exclusions if needed (e.g. `addition AND NOT hallucination AND NOT duplication`).

---

## Duplication (Linguistic conventions / Duplication)

Detects unintentional repetitions in the MT output. Three levels:
1. **Consecutive duplicate words** — "the the", "is is" (regex `\b(\w{2,})\s+\1\b`)
2. **Repeated multi-word spans** — 3–6 word n-gram appearing twice consecutively
3. **Repeated sentences** — identical sentence text appearing back-to-back

```python
import re

_DUP_WORD_RE = re.compile(r"\b(\w{2,})\s+\1\b", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def detect_duplication(text: str) -> tuple[bool, list[str]]:
    """Detect duplicated content in *text*.

    Returns:
        (flag, details) where flag is True when at least one duplication
        is found and details is a list of human-readable descriptions.
    """
    details: list[str] = []

    # 1. Consecutive duplicate words
    for m in _DUP_WORD_RE.finditer(text):
        details.append(f'repeated word: "{m.group(1)}"')

    # 2. Repeated multi-word spans (3–6 word n-grams)
    words = text.split()
    for n in range(3, min(7, len(words) // 2 + 1)):
        seen: dict[str, int] = {}
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n]).lower()
            if gram in seen and seen[gram] == i - n:
                details.append(f'repeated phrase ({n} words): "{gram}"')
            seen[gram] = i

    # 3. Repeated sentences
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) >= 2:
        prev = sentences[0].lower()
        for s in sentences[1:]:
            cur = s.lower()
            if cur == prev and len(cur) > 10:
                details.append(f'repeated sentence: "{s[:80]}"')
            prev = cur

    # Deduplicate detail messages
    seen_details: list[str] = []
    for d in details:
        if d not in seen_details:
            seen_details.append(d)
    return len(seen_details) > 0, seen_details
```

---

## Entity (Accuracy / Mistranslation / Entity)

Compares "transferable" entities between source/reference and MT:

1. **Numbers** — extracted by regex, normalised (thousand separators, decimal points), compared as sets between src ↔ mt.
2. **Identifiers** — account numbers and codes like "4567-8901" compared between src ↔ mt.
3. **Proper nouns** — capitalised non-sentence-initial words compared between ref ↔ mt (same language, directly comparable). Uses word-boundary matching to avoid false positives from sentence restructuring.

```python
_NUMBER_RE = re.compile(r"\b\d[\d',.\u2019]*\d\b|\b\d+\b")
_IDENT_RE = re.compile(r"\b[\dA-Za-z][\dA-Za-z]*[-/][\dA-Za-z-/]+\b")

def detect_entity_error(src: str, ref: str, mt: str) -> tuple[bool, list[str]]:
    """Detect entity mismatches between source/reference and MT.

    Returns (flag, details).
    """
    # Compare numbers: src vs mt (numbers should be preserved across languages)
    # Compare identifiers: src vs mt
    # Compare proper nouns: ref vs mt (same target language, word-boundary match)
    ...
```

Key design decisions:
- Numbers are compared src↔mt because numbers should transfer across languages.
- Proper nouns are compared ref↔mt (same language) to avoid cross-lingual matching issues.
- Word-boundary matching (`re.search(r"\b" + re.escape(word) + r"\b", text)`) avoids false positives when sentence structure changes.

---

## MT Hallucination (Accuracy / Mistranslation / MT hallucination)

Uses **LaBSE** cross-lingual sentence embeddings to compute cosine similarity between the source segment and its MT. When similarity falls below a threshold, the translation is flagged as "completely decoupled from the sense of the input sentence."

```python
_HALLUCINATION_THRESHOLD = 0.2

def detect_hallucination(
    srcs: list[str], mts: list[str],
    threshold: float = _HALLUCINATION_THRESHOLD,
    embed_model: str = "sentence-transformers/LaBSE",
) -> tuple[list[bool], list[float]]:
    """Detect MT hallucinations via cross-lingual embedding similarity.

    Returns (flags, scores) — flag is True when cosine similarity < threshold.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(embed_model)
    src_emb = model.encode(srcs, convert_to_numpy=True)
    mt_emb = model.encode(mts, convert_to_numpy=True)

    dot = np.sum(src_emb * mt_emb, axis=1)
    norm_src = np.linalg.norm(src_emb, axis=1)
    norm_mt = np.linalg.norm(mt_emb, axis=1)
    cos_sim = dot / (norm_src * norm_mt + 1e-9)

    flags = [bool(s < threshold) for s in cos_sim]
    scores = [round(float(s), 4) for s in cos_sim]
    return flags, scores
```

**Threshold tuning (observed on DE→EN banking/automotive TM):**
- True hallucinations: cosine similarity 0.08–0.12
- Severe omissions: 0.25–0.30
- Free translations / paraphrases: 0.60–0.80
- Correct translations: 0.75–0.99

Default threshold **0.2** catches only genuine hallucinations. Raise to 0.35 to also catch borderline cases.

---

## Addition & Omission (Accuracy / Addition, Accuracy / Omission)

Compares character-length ratio `len(mt) / len(ref)` against thresholds:
- **Addition**: ratio > 1.5 (MT is 50%+ longer than reference)
- **Omission**: ratio < 0.5 (MT is less than half the reference length)

```python
_ADDITION_RATIO = 1.5
_OMISSION_RATIO = 0.5

def detect_addition_omission(
    refs: list[str], mts: list[str],
    addition_ratio: float = _ADDITION_RATIO,
    omission_ratio: float = _OMISSION_RATIO,
) -> tuple[list[bool], list[bool], list[float]]:
    """Detect content addition and omission by length ratio.

    Returns (addition_flags, omission_flags, ratios).
    """
    add_flags, omi_flags, ratios = [], [], []
    for ref, mt in zip(refs, mts):
        ref_len = max(len(ref.strip()), 1)
        mt_len = max(len(mt.strip()), 1)
        ratio = mt_len / ref_len
        ratios.append(round(ratio, 3))
        add_flags.append(ratio > addition_ratio)
        omi_flags.append(ratio < omission_ratio)
    return add_flags, omi_flags, ratios
```

**Why length-based?** Addition and omission by definition change the amount of content. This heuristic is language-agnostic, needs no embeddings or LLM, and runs instantly. Genuine additions typically have ratios 3–6× while genuine omissions have ratios 0.2–0.4×. Hallucinations and duplications can trigger secondary addition flags (see cross-flagging note above).

---

## Grammar (Linguistic conventions / Grammar)

Uses **LanguageTool** (local Java server via `language_tool_python`) to detect grammar errors in the English MT output. The tool provides 4000+ grammar rules for English spanning subject-verb agreement, tense errors, determiner issues, pronoun case, and more.

**Installation:** `pip install language_tool_python` (downloads ~255 MB Java server on first use). Cache location: `~/.cache/language_tool_python/` — if `/home` is quota-limited, symlink this to a larger filesystem.

**Filter strategy:** Include ALL LanguageTool matches except:
- `MORFOLOGIK_RULE_*` rule IDs — pure dictionary-based spell-checker (spelling, not grammar)
- Excluded categories: `REDUNDANCY`, `STYLE`, `CASING`, `TYPOGRAPHY`, `PUNCTUATION`, `MULTITOKEN_SPELLING`, `COMPOUNDING` — these are orthographic/typographic conventions, not grammar in the MQM sense

This keeps grammar-relevant rules from categories: GRAMMAR, COLLOCATIONS, MISC, TYPOS (e.g. CONFUSION_OF_ME_I for pronoun case), NONSTANDARD_PHRASES, CONFUSED_WORDS.

```python
import language_tool_python
from typing import Any

_SPELLING_RULE_PREFIXES = ("MORFOLOGIK_RULE",)
_EXCLUDED_CATEGORIES = frozenset({
    "REDUNDANCY", "STYLE", "CASING", "TYPOGRAPHY",
    "PUNCTUATION", "MULTITOKEN_SPELLING", "COMPOUNDING",
})


def _is_grammar_match(match) -> bool:
    """Return True if a LanguageTool match is a grammar error (not spelling/style)."""
    for prefix in _SPELLING_RULE_PREFIXES:
        if match.rule_id.startswith(prefix):
            return False
    if match.category in _EXCLUDED_CATEGORIES:
        return False
    return True


def detect_grammar(text: str, tool: Any) -> tuple[bool, list[str]]:
    """Detect grammar errors in *text* using LanguageTool.

    Args:
        text: English text to check.
        tool: An initialised ``language_tool_python.LanguageTool`` instance
              (reuse across segments — starting the server is expensive).

    Returns:
        (flag, details) — flag is True when ≥1 grammar error is found;
        details is a list of human-readable descriptions.
    """
    matches = tool.check(text)
    details: list[str] = []
    for m in matches:
        if _is_grammar_match(m):
            snippet = m.matched_text or text[m.offset : m.offset + m.error_length]
            replacement = m.replacements[0] if m.replacements else "?"
            detail = (
                f'{m.category}/{m.rule_id}: "{snippet}" → "{replacement}" '
                f"({m.message})"
            )
            details.append(detail)
    return len(details) > 0, details
```

**Performance:** ~13 checks/second (single-threaded). For 1515 segments ≈ 2 minutes.

**Tested error types (all detected):**
- Subject-verb agreement (`AGREEMENT_SENT_START`)
- Double determiner (`A_MY`)
- Wrong past participle (`BEEN_PART_AGREEMENT`)
- Double negative (`DOUBLE_NEGATIVE`)
- Missing preposition (`LOOK_FORWARD_TO`)
- a/an error (`EN_A_VS_AN`)
- Pronoun case (`CONFUSION_OF_ME_I`)
- Word repetition (`ENGLISH_WORD_REPEAT_RULE`)
- Nonstandard plural — "informations" (`INFORMATIONS`)
- Pronoun-verb agreement (`HE_VERB_AGR`)
- Double comparative (`MOST_COMPARATIVE`)
- there/their confusion (`THERE_THEIR`)
- its/it's confusion (`IT_IS`)
- Modal + wrong verb form (`MD_BASEFORM`)

**Known limitations:**
- Does **not** catch noun+verb agreement like "The contract have taken" (only pronoun+verb: "He have")
- Does **not** catch calques/German-influenced word order
- Missing-comma-before-"which" is not reliably detected (ambiguous restrictive vs non-restrictive)
- Spelling errors are intentionally excluded (reserved for a separate MQM Spelling check using `MORFOLOGIK_RULE_*`)
