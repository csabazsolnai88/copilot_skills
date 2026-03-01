# MQM Heuristic Quality Checks

Six MQM error types detected without an LLM — purely via heuristics, embeddings, and local NLP tools. Each check writes a boolean flag column (`mqm_{aspect}`) for counting plus a details/score column for diagnostics. All checks use the established skip-if-exists pattern.

| MQM path | Column prefix | Method | LLM required? |
|---|---|---|---|
| Linguistic conventions / Duplication | `mqm_duplication` | Regex: consecutive repeated words, repeated n-gram spans, repeated sentences | No |
| Accuracy / Mistranslation / Entity | `mqm_entity` | NER (spaCy) — person names and org names via PER/ORG labels; src→MT and ref↔MT comparison | No |
| Accuracy / Mistranslation / MT hallucination | `mqm_hallucination` | Cross-lingual cosine similarity (LaBSE) between src ↔ mt | No (embeddings only) |
| Accuracy / Addition | `mqm_addition` | Length ratio mt/ref; flag when ratio > threshold | No |
| Accuracy / Omission | `mqm_omission` | Length ratio mt/ref; flag when ratio < threshold | No |
| Linguistic conventions / Grammar | `mqm_grammar` | LanguageTool local server — grammar rules (excluding spelling/style) | No (Java server) |

## Output columns per check

| Check | Flag column | Details column | Notes |
|---|---|---|---|
| Duplication | `mqm_duplication` (bool) | `mqm_duplication_details` (JSON list of strings) | Each string describes one duplication instance |
| Entity | `mqm_entity` (bool) | `mqm_entity_details` (JSON list of strings) | Describes missing/extra person names and org names; src→MT and ref↔MT |
| Hallucination | `mqm_hallucination` (bool) | `mqm_hallucination_score` (float 0–1) | Cosine similarity; lower = more hallucinated |
| Addition | `mqm_addition` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Omission |
| Omission | `mqm_omission` (bool) | `mqm_mt_ref_length_ratio` (float) | Shared ratio column with Addition |
| Grammar | `mqm_grammar` (bool) | `mqm_grammar_details` (JSON list of strings) + `mqm_grammar_count` (int) | Each string: `category/rule: "matched" → "fix" (message)` |

**Cross-flagging is expected.** A hallucinated segment may also be flagged for addition (different content = longer) and entity (entities don't match). A severely truncated segment may flag both omission and entity. When counting error frequencies, filter with exclusions if needed (e.g. `addition AND NOT hallucination AND NOT duplication`).

---

## Per-issue CSV exports (one row per problem)

Each boolean MQM flag column additionally produces a per-issue CSV in the output directory named after the flag (for example `mqm_entity.csv`). These CSVs are exploded so that there is exactly one row per individual problem instance. The row includes at minimum:

- `segment_id`: stable numeric identifier of the original segment (0-based)
- `src`, `trg`, `mt`: the source, reference, and MT segments (if present in the analysis table)
- additional diagnostic columns when available (e.g. `mqm_term_violations` for terminology)
- `issue`: concise issue label in the form `key:payload` where `key` is a short error class (e.g. `term_violation`, `not_in_ref`, `missing_from_mt`, `duplication`, `grammar`, `number`, `whitespace`, `untranslated`) and `payload` is a normalized short description or the problematic token/term.

Payload normalization performed before writing ensures consistent casing and removes common surrounding punctuation and diacritics (e.g. `André Helfenstein` → `andre helfenstein`). This makes downstream aggregation and filtering easier.

Note: `segment_id` is stored as an identifier and is intentionally excluded from numeric summary statistics in the analysis report.

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

MQM definition: *"Error where a name, place, or other named entity does not match the proper target language form. Example: source refers to Dublin, Ohio, but target incorrectly refers to Dublin, Ireland."*

Entity detection uses **spaCy NER** (language-appropriate model) to locate person names and organisation names.  Heuristic rules (numbers, hyphenated identifiers, capitalised words) are intentionally **not used** for this category:
- Numbers have their own ``mqm_number`` check.
- Hyphenated compound words (e.g. German *"Infotainment-System"*) are ordinary vocabulary, not named entities.
- Capitalised sentence-initial words are extremely noisy as entity signals.

Two checks are performed:

1. **Source → MT** — PER entities from the source that should appear verbatim in the MT.  Only *reliable* entities pass (``_reliable_src_entity``):
   - Multi-word names with no function/stop words and no German adjective suffixes (e.g. *"Markus Meyer"* ✓, *"Kostenlose Probefahrt"* ✗, *"Effektiver Jahreszins"* ✗).
   - All-uppercase acronyms with ≥ 2 alphabetic chars (e.g. *"BMW"*, *"AMAG"* ✓; *"E."* ✗).
   - Entities containing a digit in any token are rejected (currency formats like *"Fr. 2000.-"*).
   - Entities with short abbreviation tokens ending in `.` (≤ 3 chars) are rejected (*"B. Regulärer"*, *"Tel. P."*).
   - Entities with ` / ` separators or guillemets `«»` (cross-cell / formatting fragments) are rejected.
   - Entities matching the concatenation pattern `[a-z][A-Z]` or `[0-9][A-Z]` are rejected (missing-space NER artefacts).
   - Multi-word matching uses significant-token logic (≥ 3-char non-generic tokens) to handle translated suffixes (e.g. *"AMAG Gruppe"* → *"AMAG Group"*).
   - Note: **ORG entities are NOT included** in the src→MT check because organisation names are typically translated between languages, leading to cross-lingual false-positive rates.

2. **Reference → MT** (same-language) — PER + ORG entities in the reference but absent from MT, and extra MT entities not in reference.  Filters applied via ``_clean_ent_set``:
   - URLs (``http://``, ``https://``, ``www.``) are excluded.
   - Entities matching the concatenation pattern (case-sensitive, before lowercasing) are excluded.
   - Entities with ` / `, `, ` or guillemets `«»` are excluded (formatting fragments).
   - Entities containing any digit are excluded (currency amounts, part numbers).
   - **Single-word entities**: accepted only if all-uppercase with 2–5 alphabetic chars (e.g. *"AMAG"*, *"CUPRA"*, *"GTC"*); rejects mixed-case words, single initials, and all-caps words with >5 chars that are likely common nouns (*"ATTRAKTION"*, *"INNENRAUM"*).  Also rejects if the lowercased form is a target-language function word (e.g. *"VON"*, *"MEINE"*).
   - **Multi-word entities**: at least one word must start with uppercase in the original text.  Entities containing a target-language function/stop word are rejected (handles honorific phrases like *"Monsieur Haefner"*, *"Lieber Herr"*, slogans like *"VON EMOTIONEN GEPRÄGT"*).  German adjective inflection endings also rejected for German target (*"Maximales Drehmoment"*).  Entities with any short abbreviation token ending in `.` (≤ 3 chars) are rejected (*"M. Haefner"*, which would conflict with *"Monsieur Haefner"*).
   - Common non-entity abbreviations (currencies, role titles, generic headers) blocked via ``_NON_ENTITY_ABBREVS``.
   - MT entity superset deduplication: don't flag *"AMAG Group"* as extra when *"AMAG"* is in the reference.
   - Presence check uses ``(?<!\w)...(?!\w)`` lookarounds (not ``\b``) to correctly match entities ending in non-word chars like `.` or `&`.

**Language-specific filter word tables** (``_LANG_FILTER_WORDS``):

| Language | Covered categories |
|---|---|
| `de` | Articles, prepositions, pronouns, conjunctions, auxiliary verbs, possessive pronouns, German honorifics (*Herr, Frau, Lieber*) |
| `en` | Articles, prepositions, pronouns, conjunctions, auxiliary verbs |
| `fr` | Articles, prepositions, conjunctions, French honorifics (*Monsieur, Madame, Cher*), French sentence-initial verbs (*Profitez, Découvrir*) |
| `it` | Articles, prepositions, conjunctions |

**CLI flags:** ``--entity --src-lang de --trg-lang en``  (src-lang and trg-lang both required, default: `de`/`en`)

**Calibrated false positive rates** (on AMMT automotive/banking data):

| Language pair | Flagged | Rate |
|---|---|---|
| de→en | 55 / 1519 | 3.6% |
| de→fr | 76 / 1190 | 6.4% |
| en→de | 92 / 1674 | 5.5% |

```python
# spaCy model registry — all four must be installed:
#   pip install de-core-news-sm en-core-web-sm fr-core-news-sm it-core-news-sm
_SPACY_LANG_MODEL = {"de": "de_core_news_sm", "en": "en_core_web_sm",
                      "fr": "fr_core_news_sm", "it": "it_core_news_sm"}

_NER_TRANSFER_LABELS = {
    "de": frozenset({"PER", "ORG"}),
    "en": frozenset({"PERSON", "ORG", "PRODUCT"}),
    "fr": frozenset({"PER", "ORG"}),
    "it": frozenset({"PER", "ORG"}),
}

def detect_entity_error(
    src: str, ref: str, mt: str,
    src_lang: str = "de", trg_lang: str = "en",
) -> tuple[bool, list[str]]:
    """Detect named entity mismatches (MQM Accuracy / Mistranslation / Entity).

    Returns (flag, details).
    """
    ...
```

Key design decisions and known limitations:
- Numbers are **not** checked here — use ``--number`` for that.
- Entity presence check uses lookaround regex ``(?<!\w)...(?!\w)`` (not ``\b``) to correctly handle entities ending in non-word chars (periods, ampersands).
- Translated org suffixes (Gruppe → Group) are handled by significant-token matching for multi-word source entities.
- Single-word source entities are only checked when ALL-CAPS with 2–5 alpha chars.
- German adjective suffix regex (``_DE_ADJ_SUFFIX_RE``) covers -lich, -ig, -isch, -iv, -al, -los, -end, -haft, -bar, -sam, -voll inflection families including comparative forms (-licher, -iver, -aler).
- **Known FP: pronoun vs. name** — when the ref translator uses a pronoun (*"He"*) while MT repeats the full name, the name is flagged as an MT extra.  Not solvable without coreference resolution.
- **Known FP: committee/board naming** — *"Executive Board"* vs *"Management Board"* are lexical translation choices that may be flagged as ORG mismatches.
- **Known FP: SA vs AG** — in DE→FR, company legal-form suffix *"AG"* (German) should become *"SA"* (French); when MT keeps *"AG"*, this is flagged as a ref entity (*"SA"*) missing from MT.  This is technically a legitimate entity error.

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

---

## Number Inconsistency (Accuracy / Mistranslation / Numbers)

Detects when numeric values (integers, decimals) differ between source and MT. Numbers are extracted with a regex, normalised to floats, and compared as sets. The check flags when any number present in the source is absent from the MT, or vice versa.

```python
def detect_number_inconsistency(src: str, mt: str) -> tuple[bool, list[str]]:
    ...
```

**Number normalisation** — `_normalise_number()` handles all common thousands/decimal formats:

| Source format | Example | Normalised to |
|---|---|---|
| Swiss space-thousands (`66 900`) | `"66 900"` | `66900.0` |
| Apostrophe thousands (`66'900`) | `"66'900"` | `66900.0` |
| German dot-thousands (`30.000`) | `"30.000"` | `30000.0` |
| European decimal comma (`1,5`) | `"1,5"` | `1.5` |
| English comma-thousands (`66,900`) | `"66,900"` | `66900.0` |
| English decimal dot (`3.14`) | `"3.14"` | `3.14` |

The dot-vs-decimal ambiguity is resolved by heuristic: if **all** post-dot groups are exactly **3 digits** (e.g. `30.000`, `1.000.000`), the dot is treated as a thousands separator; otherwise it is a decimal point.

- CLI flag: `--number`
- Output columns: `mqm_number` (bool), `mqm_number_details` (JSON list)

---

## Whitespace (Linguistic conventions / Whitespace)

Detects whitespace formatting errors in the MT:

1. **Leading whitespace** — text starts with space/tab
2. **Trailing whitespace** — text ends with space/tab
3. **Double spaces / tabs** *(only if source does not have the same)* — structural formatting inherited from source is suppressed
4. **Space before period** — `" ."` pattern (after URL removal)
5. **Missing space after sentence-end punctuation** — `"word.Word"` run-on (uppercase immediately after 2+ alpha chars + period, no intervening space)

**False-positive filters applied:**
- **URLs** (`https://…`, `www.…`) are stripped from the text before checks 3–5, preventing domain-name dots from triggering.
- **Source-same double-space/tab**: skipped when the source segment has the same issue (indicates inherited structural formatting, not an MT error).
- **`_WS_MISSING_AFTER_RE`** uses a 3-char lookbehind `(?<=[alpha]{2}[.!?])` and requires the next char to be **uppercase**, so abbreviation dots (`e.g.mobility`) and bare domain names (`amag.ch`) do not fire.

```python
def detect_whitespace(text: str, src: str = "", trg_lang: str = "en") -> tuple[bool, list[str]]:
    ...
```

- CLI flag: `--whitespace`; also requires `--trg-lang {en|de|fr|it}`
- Signature now includes `src` (source segment) for source-comparison suppression.
- Output columns: `mqm_whitespace` (bool), `mqm_whitespace_details` (JSON list)

---

## Capitalization (Linguistic conventions / Capitalization)

Detects capitalization errors:

1. **Sentence-initial lowercase** — checks each sentence-initial character is uppercase.  
   Splits on `[.!?]\s+`; the following fragment is only flagged as incorrect if the **preceding fragment's last word** is NOT a known abbreviation (e.g. `e.g`, `i.e`, `etc`, `vs`, `dr`, `prof`, …) and is longer than 2 characters. This prevents false positives from `"e.g. charging cables"` and similar patterns.
2. **Lowercase first-person "I"** (English only, `trg_lang == "en"`).
3. **German noun capitalization** (German only) — uses spaCy `de_core_news_sm` to tag NOUN tokens; flags lowercase nouns.

```python
def detect_capitalization(text: str, ref: str = "", trg_lang: str = "en") -> tuple[bool, list[str]]:
    ...
```

- If a `ref` is provided, the check compares MT token casing to the reference: MT tokens that also appear in the reference but with different capitalization forms are flagged. This greatly reduces false positives by restricting checks to words present in the reference.

- CLI flag: `--capitalization`; requires `--trg-lang`
- Output columns: `mqm_capitalization` (bool), `mqm_capitalization_details` (JSON list)
- Dependency: `spacy` + `de_core_news_sm` for German noun check
- `_ABBREV_WORDS` (frozenset) contains the recognised abbreviation final-words.

---

## Unintelligible (Fluency / Unintelligible)

Detects output that is mechanically broken or incomprehensible:

1. **Unicode replacement characters** (`\ufffd`) — encoding errors
2. **Control characters** (ASCII 0–8, 14–31 excluding tab/LF/CR)
3. **Very low alphabetic ratio** — < 25% alphabetic for texts > 10 chars
4. **High non-word character ratio** — > 30% non-alphanumeric non-space chars
5. **Unexpected non-Latin script** — CJK/Arabic/Hebrew in otherwise Latin text (> 5% of text)

```python
def detect_unintelligible(text: str) -> tuple[bool, list[str]]:
    ...
```

- CLI flag: `--unintelligible`
- Output columns: `mqm_unintelligible` (bool), `mqm_unintelligible_details` (JSON list)

**Test results:** 100% F1 (20 error cases, 8 clean, 4 languages).

---

## Do-Not-Translate (Accuracy / Do-not-translate)

Checks that DNT-marked spans appear verbatim in the MT. Two annotation formats:
- `<DNT>BrandName</DNT>` (XML tag style)
- `[DNT: BrandName]` (bracket style)

```python
def detect_do_not_translate(src: str, mt: str) -> tuple[bool, list[str]]:
    ...
```

- CLI flag: `--do-not-translate`
- Output columns: `mqm_do_not_translate` (bool), `mqm_do_not_translate_details` (JSON list)
- Segments without DNT markers always return `(False, [])`.

**Test results:** 100% F1 (20 error cases, 8 clean, 4 languages).

---

## Untranslated (Accuracy / Untranslated)

Detects source-language words that were left verbatim in the MT output instead of being translated. This matches the MQM definition: "a text segment intended for translation is omitted from the target content" — for example, a German word remaining in an otherwise English translation.

The check walks content tokens (NOUN / VERB / ADJ / ADV) in the source segment using spaCy and flags any that:

1. Appear verbatim (whole-word, case-insensitive) in the MT, **and**
2. Do **not** appear in the reference translation (if the reference also kept the word, it is considered an accepted technical borrowing or cognate).

Filtered out to reduce false positives:
- Named entities (PER / ORG / PERSON) — already covered by `mqm_entity`.
- spaCy stopwords and non-content POS tags (determiners, pronouns, aux, conjunctions, …).
- Tokens shorter than 4 characters.
- Tokens containing non-alpha characters (numbers, abbreviations with punctuation, URLs).
- All-caps tokens up to 6 characters (acronyms: BMW, AMAG, …).

```python
def detect_untranslated(
    src: str,
    ref: str,
    mt: str,
    src_lang: str = "de",
    trg_lang: str = "en",   # kept for API symmetry; not used by current logic
) -> tuple[bool, list[str]]:
    ...
```

- CLI flag: `--untranslated`; requires `--src-lang` (source language) and `--trg-lang`.
- Output columns: `mqm_untranslated` (bool), `mqm_untranslated_details` (JSON list of untranslated tokens).
- Dependency: spaCy source-language model (reuses the model already loaded for `--entity`).
- The per-issue CSV `mqm_untranslated.csv` contains one row per untranslated token with `issue` label `untranslated:<token>`.

---

## Overtranslation (Accuracy / Addition / Overtranslation)

Flags MT substantially longer than reference with high novel vocabulary. Threshold: MT words > ref words × 2.5 **AND** novel vocabulary ratio > 35%.

```python
_OVERTRANSLATION_LEN_RATIO = 2.5
_OVERTRANSLATION_NOVEL_RATIO = 0.35

def detect_overtranslation(ref: str, mt: str) -> tuple[bool, list[str]]:
    ...
```

- CLI flag: `--overtranslation`
- Output columns: `mqm_overtranslation` (bool), `mqm_overtranslation_details` (JSON list)
- Note: Only catches extreme additions (2.5×); minor additions are better captured by `mqm_addition`.

**Test results:** 100% F1 (20 error cases, 8 clean, 4 languages).

---

## Undertranslation (Accuracy / Omission / Undertranslation)

Flags MT substantially shorter than reference with poor vocabulary coverage. Threshold: MT words < ref words × 0.65 **AND** ref vocabulary coverage < 55%.

```python
_UNDERTRANSLATION_LEN_RATIO = 0.65
_UNDERTRANSLATION_COVERAGE = 0.55

def detect_undertranslation(ref: str, mt: str) -> tuple[bool, list[str]]:
    ...
```

- Minimum reference length: 5 words.
- CLI flag: `--undertranslation`
- Output columns: `mqm_undertranslation` (bool), `mqm_undertranslation_details` (JSON list)
- Note: Only catches severe truncations (< 65% length); minor omissions are better captured by `mqm_omission`.

**Test results:** 100% F1 (20 error cases, 8 clean, 4 languages).
