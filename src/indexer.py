"""
indexer.py — Inverted-index builder for COMP3011 Coursework 2

Accepts the list of page dicts produced by crawler.crawl() and builds an
inverted index mapping every token to its per-page frequency and positions.

Data structure
--------------
The index is a dict with two kinds of top-level keys:

  "_meta"   – A reserved key (underscore-prefixed so it can never collide
               with a real word token).  Maps each page URL to a dict
               containing ``total_tokens``: the number of tokens on that
               page after tokenisation and punctuation stripping.  This is
               needed later by search.py to compute
               TF = frequency / total_tokens.

  <word>    – Every other top-level key is a word.  Each maps URLs to a
               dict of ``frequency`` (int) and ``positions`` (list of
               0-based ints).

Tokenisation rules (applied in order)
--------------------------------------
1. Split on whitespace.
2. Strip punctuation from the *edges* of each token only
   (``str.strip(string.punctuation)``).
3. Discard any tokens that become empty strings after stripping.
4. No stopword removal — every surviving word is indexed.
5. No stemming — words are stored in their original lowercased form
   (the crawler already lowercases text, so we do not lowercase again).
"""

import json
import os
import string


# ── Private helpers ──────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    """Tokenise a page's text according to the agreed rules.

    1. Split on whitespace.
    2. Strip leading/trailing punctuation from each token.
    3. Discard any resulting empty strings.

    Returns a list of tokens preserving order (needed for position tracking).
    """
    raw_tokens = text.split()
    tokens = []
    for raw in raw_tokens:
        stripped = raw.strip(string.punctuation)
        if stripped:                          # discard empty strings
            tokens.append(stripped)
    return tokens


# ── Public API ───────────────────────────────────────────────────────────────


def build_index(pages: list[dict]) -> dict:
    """Build and return the inverted index from a list of crawled page dicts.

    Each element of *pages* must have keys ``"url"`` (str) and ``"text"``
    (str).  The returned dict always contains a ``"_meta"`` key with
    per-page token counts, plus one key per unique word found across all
    pages.
    """
    # Initialise the index with an empty _meta section.
    index: dict = {"_meta": {}}

    for page in pages:
        url = page["url"]
        text = page["text"]

        tokens = _tokenise(text)

        # ── Store per-page metadata ──────────────────────────────────
        # total_tokens is required by search.py to compute
        # TF = frequency / total_tokens at query time.
        index["_meta"][url] = {"total_tokens": len(tokens)}

        # ── Build word → URL → {frequency, positions} mappings ───────
        for position, token in enumerate(tokens):
            if token not in index:
                index[token] = {}

            if url not in index[token]:
                index[token][url] = {"frequency": 0, "positions": []}

            index[token][url]["frequency"] += 1
            index[token][url]["positions"].append(position)

    return index


def save_index(index: dict, filepath: str = "data/index.json") -> None:
    """Save the index to disk as JSON.  Creates the directory if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)


def load_index(filepath: str = "data/index.json") -> dict:
    """Load and return the index from a JSON file on disk.

    Raises ``FileNotFoundError`` if *filepath* does not exist.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)
