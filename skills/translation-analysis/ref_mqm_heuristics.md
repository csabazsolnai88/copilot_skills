# MQM Heuristic Quality Checks

Five MQM error types detected without an LLM — purely via heuristics and embeddings. Each check writes two columns: a boolean flag (`mqm_{aspect}`) for counting and a details/score column for diagnostics. All checks run unconditionally (no `--termbase` or LLM required) and use the established skip-if-exists pattern.

| MQM path | Column prefix | Method | LLM required? |
|---|---|---|---|
| Linguistic conventions / Duplication | `mqm_duplication` | Regex: consecutive repeated words, repeated n-gram spans, repeated sentences | No |
| Accuracy / Mistranslation / Entity | `mqm_entity` | Regex + set comparison: numbers, identifiers, proper nouns | No |
| Accuracy / Mistranslation / MT hallucination | `mqm_hallucination` | Cross-lingual cosine similarity (LaBSE) between src ↔ mt | No (embeddings only) |
| Accuracy / Addition | `mqm_addition` | Length ratio mt/ref; flag when ratio > threshold | No |
| Accuracy / Omission | `mqm_omission` | Length ratio mt/ref; flag when ratio < threshold | No |

## Output columns per check

| Check | Flag column | Details column | Notes |
|---|---|---|---|
| Duplication | `mqm_duplication` (bool) | `mqm_duplication_details` (JSON list of strings) | Each string describes one duplication instance |
| Entity | `mqm_entity` (bool) | `mqm_entity_details` (JSON list of strings) | Describes missing/extra numbers, identifiers, proper nouns |
| Hallucination | `mqm_hallucination` (bool) | `mqm_hallucination_score` (float 0–1) | Cosine similarity; lower = more hallucinated |
| Addition | `mqm_addition` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Omission |
| Omission | `mqm_omission` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Addition |

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
