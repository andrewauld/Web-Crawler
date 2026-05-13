"""
search.py — Query-time search logic for COMP3011 Coursework 2

Provides five public functions:

  print_word   – look up a single word's postings list
  compute_tfidf – TF-IDF score for one word on one page
  search_and   – AND search across multiple terms (intersection)
  search_or    – OR  search across multiple terms (union)
  suggest      – find morphologically related vocabulary words via stemming

The module consumes the inverted index produced by indexer.py.  The index
is expected to have a ``_meta`` key with per-page token counts and one key
per word, each mapping URLs to ``{frequency, positions}`` dicts.
"""

import math

import nltk
from nltk.stem import SnowballStemmer

# Ensure the punkt tokeniser data is available (required by NLTK internals).
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Initialise the stemmer once at module level for reuse.
_stemmer = SnowballStemmer("english")


# ── Private helpers ──────────────────────────────────────────────────────────


def _get_urls(index: dict, word: str) -> set[str]:
    """Return the set of URLs that contain *word*, or an empty set."""
    if word in index:
        return set(index[word].keys())
    return set()


# ── Public API ───────────────────────────────────────────────────────────────


def print_word(index: dict, word: str) -> dict | None:
    """Return the postings dict for a word, or None if not found.

    Lowercases the query word before lookup.
    """
    word = word.lower()
    if word in index and word != "_meta":
        return index[word]
    return None


def compute_tfidf(index: dict, word: str, url: str) -> float:
    """Compute the TF-IDF score for a given word on a given page.

    Formula
    -------
    TF(word, page)     = frequency(word, page) / total_tokens(page)
    IDF(word)          = ln(total_pages / pages_containing_word)
    TF-IDF(word, page) = TF * IDF

    Edge cases
    ----------
    - Word not in index         → 0.0
    - URL not in word's postings → 0.0
    - total_tokens == 0         → 0.0
    - total_pages  == 0         → 0.0

    Returns 0.0 for any edge case rather than raising.
    """
    # Number of indexed pages.
    total_pages = len(index.get("_meta", {}))
    if total_pages == 0:
        return 0.0

    # The word must exist in the index (and not be _meta).
    if word not in index or word == "_meta":
        return 0.0

    postings = index[word]

    # The URL must appear in the word's postings.
    if url not in postings:
        return 0.0

    frequency = postings[url]["frequency"]

    # Retrieve total_tokens for this page from _meta.
    total_tokens = index["_meta"].get(url, {}).get("total_tokens", 0)
    if total_tokens == 0:
        return 0.0

    # TF = frequency / total_tokens
    tf = frequency / total_tokens

    # IDF = ln(total_pages / pages_containing_word)
    pages_containing_word = len(postings)
    idf = math.log(total_pages / pages_containing_word)

    return tf * idf


def search_and(index: dict, query_terms: list[str]) -> list[tuple[str, float]]:
    """Return pages containing ALL query terms, ranked by aggregate TF-IDF.

    Algorithm
    ---------
    1. Lowercase all query terms.
    2. Retrieve URL sets for each term.
    3. Intersect all URL sets — pages must contain every term.
    4. For each URL in the intersection, sum TF-IDF scores across all terms.
    5. Return (url, score) tuples sorted by score descending.

    Returns an empty list when the intersection is empty or no terms given.
    """
    if not query_terms:
        return []

    terms = [t.lower() for t in query_terms]

    # Collect per-term URL sets and intersect.
    url_sets = [_get_urls(index, term) for term in terms]
    common_urls = url_sets[0]
    for s in url_sets[1:]:
        common_urls = common_urls & s

    if not common_urls:
        return []

    # Score each URL by summing TF-IDF across all query terms.
    results: list[tuple[str, float]] = []
    for url in common_urls:
        score = sum(compute_tfidf(index, term, url) for term in terms)
        results.append((url, score))

    # Sort by score descending.
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def search_or(index: dict, query_terms: list[str]) -> list[tuple[str, float]]:
    """Return pages containing ANY query term, ranked by aggregate TF-IDF.

    Algorithm
    ---------
    1. Lowercase all query terms.
    2. Retrieve URL sets for each term.
    3. Union all URL sets.
    4. For each URL, sum TF-IDF scores for the terms that appear on that page
       (a term absent from a page contributes 0.0).
    5. Return (url, score) tuples sorted by score descending.

    Returns an empty list when no term exists in the index or no terms given.
    """
    if not query_terms:
        return []

    terms = [t.lower() for t in query_terms]

    # Collect per-term URL sets and union.
    all_urls: set[str] = set()
    for term in terms:
        all_urls |= _get_urls(index, term)

    if not all_urls:
        return []

    # Score each URL by summing TF-IDF across all query terms.
    # compute_tfidf already returns 0.0 when a word is absent from a page.
    results: list[tuple[str, float]] = []
    for url in all_urls:
        score = sum(compute_tfidf(index, term, url) for term in terms)
        results.append((url, score))

    # Sort by score descending.
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def suggest(index: dict, query_word: str) -> list[str]:
    """Return vocabulary words in the index that share a stem with *query_word*.

    Uses the NLTK Snowball stemmer (English).  The index itself is **not**
    stemmed — full word forms are preserved.

    - Stems the query word.
    - Iterates over all index keys, skipping ``_meta``.
    - Collects keys whose stem matches the query stem.
    - Excludes the query word itself.
    - Returns an alphabetically sorted list.
    """
    query_lower = query_word.lower()
    query_stem = _stemmer.stem(query_lower)

    related: list[str] = []
    for key in index:
        if key == "_meta":
            continue
        if key == query_lower:
            continue  # exclude the query word itself
        if _stemmer.stem(key) == query_stem:
            related.append(key)

    related.sort()
    return related
