# Topic Modeling — LDA & BERTopic

Topic modeling helps group segments by domain and surface topic-specific MT issues. The workflow mirrors other analyses: (1) discover topics on a representative corpus, (2) compute per-segment topic diagnostics and append useful columns to the output table. Do NOT add categorical topic-id columns for aggregation (e.g. `*_topic_id`) — topic ids are labels and are not meaningful to average or aggregate numerically. Recommended per-model output columns: `{prefix}_topic_score`, `{prefix}_topic_terms` (and report topic distributions, medians, or score statistics when summarizing topics).

Two complementary approaches are provided below. Both are tuned for **short TM segments** in a specialised domain (e.g. banking/insurance, automotive) and support **multiple languages** (tested on English, German, French, and Italian). Adapt the stop-word sets and parameters for other domains.

---

## Shared helpers (place near the top of the script)

```python
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Regex that removes common TM placeholders ({1}, &1, #124, #AUX) and bare numbers
# before feeding text to a bag-of-words model.
_PLACEHOLDER_RE = re.compile(r"\{[^}]*\}|&\w+|#[\w-]+|\b\d+\b")

# ---------------------------------------------------------------------------
# Extra stop words — English
# ---------------------------------------------------------------------------

# Words that leak into topic labels but carry no topic signal in banking TM.
_EXTRA_STOPS_EN: frozenset[str] = frozenset({
    "please", "note", "via", "using", "used", "use",
    "carried", "does", "set", "given", "trigger", "opened", "open",
    "aux",
    "greater", "less", "pending", "required", "minimum", "maximum",
    "linked", "means", "following", "date", "end",
    "number", "type", "new",
    # "valid" generates same-word bigrams ("valid valid") on short TM segments
    "valid",
})

# Lighter English stop list for BERTopic: c-TF-IDF handles cross-topic
# discrimination; keep cluster-discriminative terms like "valid", "date".
_BERTOPIC_EXTRA_STOPS_EN: frozenset[str] = frozenset({
    "please", "note", "via", "using", "used", "use",
    "carried", "does", "set", "given", "trigger", "opened", "open",
    "aux", "linked", "means", "number", "type", "new",
})

# ---------------------------------------------------------------------------
# Extra stop words — German
# ---------------------------------------------------------------------------

# German words that slip through spacy's stop-word list in banking TM segments.
_EXTRA_STOPS_DE: frozenset[str] = frozenset({
    "bitte", "hinweis", "aux", "gilt", "sofern", "erfolgt",
    "gemäss", "gemäß", "entsprechend", "folgende", "folgenden",
    "neu", "neue", "neuen", "typ",
})

# Lighter German stop list for BERTopic (keep "gültig", "datum" etc.).
_BERTOPIC_EXTRA_STOPS_DE: frozenset[str] = frozenset({
    "bitte", "hinweis", "aux", "sofern",
    "gemäss", "gemäß", "entsprechend", "folgende", "folgenden",
})

# ---------------------------------------------------------------------------
# Extra stop words — French
# ---------------------------------------------------------------------------

# French words that slip through spacy's stop-word list in automotive TM segments.
_EXTRA_STOPS_FR: frozenset[str] = frozenset({
    "veuillez", "merci", "svp", "via", "aux", "puis",
    "nouveau", "nouvelle", "nouveaux", "nouvelles",
    "type", "voir", "selon", "suite",
    # generic time / discourse words
    "temps", "dernière", "dernier", "janvier", "février", "mars",
    "également", "été", "fait", "fois", "cas", "bien", "très",
    "agit", "permet", "ainsi", "encore", "lors", "toujours",
    "donc", "part", "aussi", "déjà", "car", "grâce",
})

_BERTOPIC_EXTRA_STOPS_FR: frozenset[str] = frozenset({
    "veuillez", "merci", "svp", "via", "aux", "puis",
    "voir", "selon",
    "temps", "dernière", "dernier", "également", "été",
    "fait", "fois", "cas", "bien", "très",
    "agit", "permet", "ainsi", "encore", "lors", "toujours",
    "donc", "part", "aussi", "déjà", "car", "grâce",
})

# ---------------------------------------------------------------------------
# Extra stop words — Italian
# ---------------------------------------------------------------------------

# Italian words that slip through spacy's stop-word list in automotive TM segments.
_EXTRA_STOPS_IT: frozenset[str] = frozenset({
    "prego", "cortesemente", "gentilmente", "tramite", "aux",
    "nuovo", "nuova", "nuovi", "nuove",
    "tipo", "vedere", "secondo", "seguito",
    "vuole", "gennaio", "febbraio", "marzo",
    "possibile", "seguenti", "propria", "proprio",
    "completamente", "numerosi", "ancora", "sempre",
    "ogni", "inoltre", "quindi", "già", "solo",
    "fatto", "stata", "stato", "modo", "volta",
})

_BERTOPIC_EXTRA_STOPS_IT: frozenset[str] = frozenset({
    "prego", "cortesemente", "gentilmente", "tramite", "aux",
    "vedere", "secondo",
    "vuole", "possibile", "seguenti", "propria", "proprio",
    "completamente", "numerosi", "ancora", "sempre",
    "ogni", "inoltre", "quindi", "già", "solo",
    "fatto", "stata", "stato", "modo", "volta",
})

# ---------------------------------------------------------------------------
# Stop-word config & helper
# ---------------------------------------------------------------------------

# Mapping: lang code → (spacy_model, nltk_name, extra_full, extra_bt)
_STOP_WORD_CONFIG: dict[str, tuple[str, str, frozenset, frozenset]] = {
    "de": ("de_core_news_sm", "german",  _EXTRA_STOPS_DE, _BERTOPIC_EXTRA_STOPS_DE),
    "fr": ("fr_core_news_sm", "french",  _EXTRA_STOPS_FR, _BERTOPIC_EXTRA_STOPS_FR),
    "it": ("it_core_news_sm", "italian", _EXTRA_STOPS_IT, _BERTOPIC_EXTRA_STOPS_IT),
}

def _get_stop_words(lang: str, for_bertopic: bool = False) -> list[str]:
    """Return a merged stop-word list for the given language.

    For English: uses sklearn's ENGLISH_STOP_WORDS.
    For German/French/Italian: uses spacy's stop-word list (covers modal/
    auxiliary verbs that NLTK misses) plus domain extras.  Requires spacy
    and the corresponding model; falls back to NLTK if spacy is unavailable.

    Supported spacy models:
      - de: de_core_news_sm (543 words)
      - fr: fr_core_news_sm (507 words)
      - it: it_core_news_sm (624 words)

    Args:
        lang:         ISO 639-1 code ("en", "de", "fr", "it").
        for_bertopic: If True, use the lighter stop list (BERTopic's c-TF-IDF
                      handles discrimination; only remove generic UI noise).

    Returns:
        Unique stop words as a plain ``list`` (required by sklearn CountVectorizer).
    """
    cfg = _STOP_WORD_CONFIG.get(lang)
    if cfg is not None:
        spacy_model, nltk_name, extra_full, extra_bt = cfg
        try:
            import spacy
            nlp = spacy.load(spacy_model)
            base: set[str] = {w.lower() for w in nlp.Defaults.stop_words}
        except Exception:
            from nltk.corpus import stopwords as _nltk
            base = set(_nltk.words(nltk_name))
        extra = extra_bt if for_bertopic else extra_full
    else:  # default: English
        base = set(ENGLISH_STOP_WORDS)
        extra = _BERTOPIC_EXTRA_STOPS_EN if for_bertopic else _EXTRA_STOPS_EN
    return list(base | extra)


def _clean_for_lda(text: str, lang: str = "en") -> str:
    """Strip TM placeholders and normalise text before vectorisation.

    For English removes all non-ASCII letters.
    For German and other languages preserves Unicode word characters so that
    umlauts (ä, ö, ü) and ß survive cleaning.

    Args:
        text: Raw segment text.
        lang: ISO 639-1 language code.
    """
    text = _PLACEHOLDER_RE.sub(" ", text)
    if lang == "en":
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    else:
        # Keep Unicode letters; remove digits and punctuation
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
```

---

## Example A — LDA (bag-of-words, interpretable, good for ≥ 100 docs)

LDA with bigrams and a language-aware stop-word list. Works well when you need
fast, explainable topics without GPU/embedding dependencies.

Key tuning knobs:
- **`n_topics`** — start at 5–8 for a domain-specific TM; increase until topics look distinct.
- **`doc_topic_prior` (α)** — low value (0.1) forces each document into fewer topics, producing sharper assignments.
- **`topic_word_prior` (β)** — low value (0.01) forces each topic to concentrate on fewer words, giving cleaner labels.

```python
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def run_lda(
    docs: list[str],
    n_topics: int = 6,
    n_top_terms: int = 8,
    lang: str = "en",
) -> tuple[list[int], list[float], list[str], dict[int, str]]:
    """Fit LDA on *docs* and return per-document topic assignments.

    Args:
        docs:        Raw text documents (one per TM segment).
        n_topics:    Number of topics (K).
        n_top_terms: Number of top terms to include in each topic label.
        lang:        ISO 639-1 code of the document language ("en", "de", …).
                     Controls stop-word list and tokenisation pattern.

    Returns:
        (topic_ids, topic_scores, topic_terms_per_doc, topic_label_map)
        where topic_label_map maps topic int → comma-separated term string.
    """
    cleaned = [_clean_for_lda(d, lang=lang) for d in docs]
    stop_words = _get_stop_words(lang, for_bertopic=False)

    # CountVectorizer settings tuned for short domain-specific TM segments.
    # token_pattern uses Unicode \w so German umlauts are tokenised correctly.
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=2,          # require term in ≥ 2 docs (removes hapaxes)
        max_df=0.85,       # ignore terms in > 85 % of docs (near-constant words)
        stop_words=stop_words,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",  # min 3-char Unicode letters
        max_features=300,
    )
    X = vectorizer.fit_transform(cleaned)

    # Remove same-word bigrams ("valid valid", "gültig gültig") that appear on
    # very short or repetitive segments and add noise to topic representations.
    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
    keep_mask = np.array(
        [len(f.split()) < 2 or f.split()[0] != f.split()[1] for f in feature_names]
    )
    X = X[:, keep_mask]
    feature_names = [f for f, keep in zip(feature_names, keep_mask) if keep]

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        random_state=42,
        n_jobs=-1,             # parallelise E-step across all cores
        doc_topic_prior=0.1,   # low α → sharper per-doc topic assignments
        topic_word_prior=0.01, # low β → sharper per-topic word distributions
    )
    theta = lda.fit_transform(X)  # shape (n_docs, n_topics)

    topic_label_map: dict[int, str] = {}
    for k in range(n_topics):
        top_idx = lda.components_[k].argsort()[-(n_top_terms):][::-1]
        topic_label_map[k] = ", ".join(feature_names[i] for i in top_idx)

    topic_ids    = [int(row.argmax())  for row in theta]
    topic_scores = [float(row.max())   for row in theta]
    topic_terms  = [topic_label_map[t] for t in topic_ids]

    return topic_ids, topic_scores, topic_terms, topic_label_map


# --- usage ---
# EN:  topic_ids, topic_scores, topic_terms, _ = run_lda(docs, n_topics=6, lang="en")
# DE:  topic_ids, topic_scores, topic_terms, _ = run_lda(docs, n_topics=6, lang="de")
# FR:  topic_ids, topic_scores, topic_terms, _ = run_lda(docs, n_topics=6, lang="fr")
# IT:  topic_ids, topic_scores, topic_terms, _ = run_lda(docs, n_topics=6, lang="it")
```

---

## Example B — BERTopic (embedding + clustering; good for short / multilingual segments)

BERTopic uses sentence embeddings + UMAP + HDBSCAN. Better than LDA for short,
noisy segments because semantic similarity is captured in the embedding space.

Key tuning knobs:
- **`min_cluster_size`** — HDBSCAN parameter. Lower → more topics, fewer outliers.
  For < 200 docs, try 3–6. Segments assigned to no cluster get topic id `-1` (outliers).
- **`n_neighbors` (UMAP)** — local neighbourhood size. Smaller values preserve local
  structure; sensible range 5–15 for < 200 docs.
- **`vectorizer_model`** — pass a `CountVectorizer` with your domain stop words so
  BERTopic's c-TF-IDF term labels are clean. Use `min_df=1` to include rare but
  cluster-specific terms (unlike LDA, c-TF-IDF handles discrimination via IDF).
- **`language` parameter** — **must** be `"multilingual"` for non-English text. When
  `language="english"` BERTopic internally strips `[^A-Za-z0-9 ]` via
  `_preprocess_text`, which destroys German umlauts (ä→dropped, ö→dropped, etc.).

Embedding model selection:
- English → `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim)
- All other languages → `sentence-transformers/LaBSE` (109 languages, 768-dim,
  cached at `/scratch/csaba/HF_model_cache` on this server)

```python
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Auto-select embedding model by language
_DEFAULT_BERTOPIC_MODEL: dict[str, str] = {
    "en": "sentence-transformers/all-MiniLM-L6-v2",
    "default": "sentence-transformers/LaBSE",  # 109 languages
}


def run_bertopic(
    docs: list[str],
    n_top_terms: int = 8,
    min_cluster_size: int = 4,
    embed_model: str | None = None,
    lang: str = "en",
) -> tuple[list[int], list[float], list[str], dict[int, str]]:
    """Fit BERTopic on *docs* and return per-document topic assignments.

    For small corpora (< 200 docs) UMAP and HDBSCAN parameters are tuned
    to produce smaller, tighter clusters instead of lumping everything into
    the outlier bucket (-1).

    IMPORTANT: pass ``lang`` correctly. BERTopic applies ASCII-only preprocessing
    when ``language="english"``, which destroys non-ASCII characters (umlauts,
    accents, etc.). For non-English input use any other language string.

    Args:
        docs:             Raw text documents.
        n_top_terms:      Number of top terms per topic label.
        min_cluster_size: HDBSCAN minimum cluster size. Lower = more clusters,
                          fewer outliers. Sensible range: 3–8 for small corpora.
        embed_model:      SentenceTransformer model name override.  If None,
                          auto-selected: English → all-MiniLM-L6-v2,
                          others → LaBSE.
        lang:             ISO 639-1 code ("en", "de", …). Controls stop-word
                          list, embedding model, and BERTopic language setting.

    Returns:
        (topic_ids, topic_scores, topic_terms_per_doc, topic_label_map)
        Topic -1 means the segment is an outlier (not assigned to any cluster).
    """
    if embed_model is None:
        embed_model = _DEFAULT_BERTOPIC_MODEL.get(lang, _DEFAULT_BERTOPIC_MODEL["default"])

    cleaned = [_clean_for_lda(d, lang=lang) for d in docs]

    # Custom vectorizer: use the lighter BERTopic stop list so cluster-
    # discriminative terms (e.g. "valid"/"gültig", "date"/"datum") remain
    # visible as labels. min_df=1 ensures rare cluster-unique terms like
    # "bollo" or "ticino" appear in the label of their specialist cluster.
    # Use Unicode token_pattern so umlauts are tokenised correctly.
    stop_words = _get_stop_words(lang, for_bertopic=True)
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
    )

    embed = SentenceTransformer(embed_model)
    embeddings = embed.encode(cleaned, show_progress_bar=True, convert_to_numpy=True)

    # UMAP tuned for small corpora:
    #   n_neighbors=5  — tight local neighbourhood; good for < 100 docs
    #   n_components=5 — latent dim before HDBSCAN
    #   min_dist=0.05  — keep similar docs tightly packed
    umap_model = UMAP(
        n_neighbors=5,
        n_components=5,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN: min_samples=1 makes HDBSCAN less conservative about outliers.
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # CRITICAL: use language="multilingual" for non-English text.
    # When language="english", BERTopic's _preprocess_text applies
    # re.sub(r"[^A-Za-z0-9 ]+", "", doc) *internally*, stripping umlauts
    # and all non-ASCII characters before c-TF-IDF term extraction.
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="english" if lang == "en" else "multilingual",
        calculate_probabilities=True,
        verbose=False,
        top_n_words=n_top_terms,
    )
    topics, probs = topic_model.fit_transform(cleaned, embeddings)

    def _get_terms(t: int) -> str:
        if t == -1:
            return "(outlier)"
        words = topic_model.get_topic(t)
        return ", ".join(w for w, _ in words if w.strip()) if words else "(no terms)"

    topic_label_map = {t: _get_terms(t) for t in set(topics)}
    topic_scores = [float(p.max()) if hasattr(p, "max") else 0.0 for p in probs]
    topic_terms  = [topic_label_map[t] for t in topics]

    return list(topics), topic_scores, topic_terms, topic_label_map


# --- usage ---
# EN:  bt_ids, bt_scores, bt_terms, _ = run_bertopic(docs, lang="en")
# DE:  bt_ids, bt_scores, bt_terms, _ = run_bertopic(docs, lang="de")
# FR:  bt_ids, bt_scores, bt_terms, _ = run_bertopic(docs, min_cluster_size=5, lang="fr")
# IT:  bt_ids, bt_scores, bt_terms, _ = run_bertopic(docs, min_cluster_size=5, lang="it")
```

## Notes

- Output columns: `{prefix}_topic_score`, `{prefix}_topic_terms` per model.
    When running topic modeling on both source (DE) and target (EN), use a column prefix
    to keep them separate, e.g. `lda_src_topic_score` vs `lda_topic_score`.
    Note: Do NOT create categorical `{prefix}_topic_id` columns for aggregation — topic ids
    are labels and aggregating them (mean/median) is not meaningful. If you need a
    corpus-level summary, report topic-score statistics or topic distributions instead
    (e.g. `{prefix}_topic_score_mean`).
- **Skip-if-exists pattern**: check `set(df.columns)` before running each model so the
  script can be re-run incrementally without recomputing already-present columns.
- **BERTopic outliers** (topic id `-1`): typically 5–15 % of segments on small datasets.
  Reduce `min_cluster_size` if you want fewer outliers (at the cost of noisier clusters).
- **Non-English stop words**: spacy models are strongly preferred over NLTK because they
  cover modal/auxiliary verbs that NLTK misses:
  - German: `de_core_news_sm` (543 words) vs NLTK (232 words)
  - French: `fr_core_news_sm` (507 words) vs NLTK (157 words)
  - Italian: `it_core_news_sm` (624 words) vs NLTK (279 words)
  Use `_STOP_WORD_CONFIG` dict to add new languages — just provide the spacy model name,
  NLTK corpus name, and two extra stop-word frozensets.
- **LaBSE** (`sentence-transformers/LaBSE`) is an excellent multilingual embedding model
  covering 109 languages. For this server it is cached at `/scratch/csaba/HF_model_cache`.
  Used automatically for any non-English language.
- For large corpora (> 10k docs) prefer LDA or mini-batch HDBSCAN; BERTopic/UMAP can be
  slow at scale.
