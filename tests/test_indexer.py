"""
test_indexer.py — pytest suite for the inverted-index builder

Covers every test case specified in the coursework brief:
  • Correct index structure
  • _meta total_tokens accuracy
  • _meta present on empty page
  • Frequency counting
  • Position tracking
  • Multi-page indexing
  • Word unique to one page
  • Punctuation stripping
  • Empty page (no crash, no word entries)
  • Empty input list
  • Save to JSON (tmp_path)
  • Load from JSON (tmp_path)
  • Load missing file raises FileNotFoundError
  • Round-trip fidelity (save → load)
"""

import json
import os
import sys

import pytest

# Ensure the src package is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import build_index, save_index, load_index


# ── Helpers ──────────────────────────────────────────────────────────────────

def _single_page(text: str, url: str = "https://example.com/") -> list[dict]:
    """Convenience: wrap a text string in the page-dict format."""
    return [{"url": url, "text": text}]


# ── Structure ────────────────────────────────────────────────────────────────


class TestIndexStructure:
    """Verify the top-level shape of the returned index."""

    def test_correct_index_structure(self):
        """The index must contain '_meta' and word keys with the agreed
        nested shape: word → url → {frequency, positions}."""
        pages = _single_page("the cat sat on the mat")
        index = build_index(pages)

        # _meta must be present
        assert "_meta" in index

        # Every non-_meta key should map URLs to {frequency, positions}
        for key, url_map in index.items():
            if key == "_meta":
                continue
            for url, stats in url_map.items():
                assert "frequency" in stats
                assert "positions" in stats
                assert isinstance(stats["frequency"], int)
                assert isinstance(stats["positions"], list)


# ── _meta ────────────────────────────────────────────────────────────────────


class TestMeta:
    """Tests for the _meta per-page metadata section."""

    def test_total_tokens_accuracy(self):
        """total_tokens must equal the exact token count after punctuation
        stripping for a known input string."""
        # "hello, world!" → tokens ["hello", "world"] → 2
        pages = _single_page("hello, world!")
        index = build_index(pages)
        assert index["_meta"]["https://example.com/"]["total_tokens"] == 2

    def test_meta_present_on_empty_page(self):
        """A page whose text yields zero tokens must still have a _meta
        entry with total_tokens: 0."""
        # Text that consists entirely of punctuation
        pages = _single_page("... !!! ???")
        index = build_index(pages)
        assert index["_meta"]["https://example.com/"]["total_tokens"] == 0


# ── Frequency & positions ───────────────────────────────────────────────────


class TestFrequencyAndPositions:
    """Validate per-word frequency counts and position lists."""

    def test_frequency_counting(self):
        """A word appearing N times on a page must have frequency N."""
        pages = _single_page("go go go")
        index = build_index(pages)
        assert index["go"]["https://example.com/"]["frequency"] == 3

    def test_position_tracking(self):
        """Positions must be correct 0-indexed integers."""
        pages = _single_page("alpha beta alpha")
        index = build_index(pages)
        entry = index["alpha"]["https://example.com/"]
        assert entry["positions"] == [0, 2]
        assert entry["frequency"] == 2

        entry_beta = index["beta"]["https://example.com/"]
        assert entry_beta["positions"] == [1]
        assert entry_beta["frequency"] == 1


# ── Multi-page scenarios ────────────────────────────────────────────────────


class TestMultiPage:
    """Tests that span more than one crawled page."""

    def test_multi_page_indexing(self):
        """A word appearing on two pages must have two URL entries with
        independent frequency and position stats."""
        pages = [
            {"url": "https://example.com/a", "text": "hello world"},
            {"url": "https://example.com/b", "text": "world peace"},
        ]
        index = build_index(pages)

        # "world" appears on both pages
        assert "https://example.com/a" in index["world"]
        assert "https://example.com/b" in index["world"]

        # Each page has independent stats
        assert index["world"]["https://example.com/a"]["frequency"] == 1
        assert index["world"]["https://example.com/a"]["positions"] == [1]
        assert index["world"]["https://example.com/b"]["frequency"] == 1
        assert index["world"]["https://example.com/b"]["positions"] == [0]

    def test_word_unique_to_one_page(self):
        """A word that appears on only one page must have exactly one URL
        entry in its mapping."""
        pages = [
            {"url": "https://example.com/a", "text": "unique common"},
            {"url": "https://example.com/b", "text": "common"},
        ]
        index = build_index(pages)

        assert len(index["unique"]) == 1
        assert "https://example.com/a" in index["unique"]


# ── Tokenisation edge cases ─────────────────────────────────────────────────


class TestTokenisation:
    """Verify punctuation stripping and edge-case handling."""

    def test_punctuation_stripping(self):
        """'hello,' and 'hello' must be indexed under the same token."""
        pages = _single_page("hello, hello")
        index = build_index(pages)

        assert "hello" in index
        assert index["hello"]["https://example.com/"]["frequency"] == 2
        assert index["hello"]["https://example.com/"]["positions"] == [0, 1]

        # No stray punctuated variant should exist
        assert "hello," not in index

    def test_empty_page(self):
        """A page with no usable tokens must not crash and must contribute
        nothing to word entries (only _meta)."""
        pages = _single_page(",,, ... !!!")
        index = build_index(pages)

        # _meta entry must still be present
        assert "https://example.com/" in index["_meta"]
        assert index["_meta"]["https://example.com/"]["total_tokens"] == 0

        # The only key in the entire index should be _meta
        assert list(index.keys()) == ["_meta"]

    def test_empty_input(self):
        """build_index([]) must return {'_meta': {}} without error."""
        index = build_index([])
        assert index == {"_meta": {}}


# ── Persistence (save / load / round-trip) ───────────────────────────────────


class TestPersistence:
    """File I/O tests using pytest's tmp_path fixture — no real data/ writes."""

    def test_save_to_json(self, tmp_path):
        """save_index must create the file and its contents must deserialise
        back to the original index."""
        index = build_index(_single_page("save me"))
        filepath = str(tmp_path / "index.json")

        save_index(index, filepath)

        assert os.path.isfile(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded == index

    def test_save_creates_directory(self, tmp_path):
        """save_index must create intermediate directories if they do not
        already exist."""
        filepath = str(tmp_path / "nested" / "dir" / "index.json")
        index = build_index(_single_page("nested"))

        save_index(index, filepath)

        assert os.path.isfile(filepath)

    def test_load_from_json(self, tmp_path):
        """load_index must correctly deserialise a known JSON file."""
        expected = {"_meta": {}, "hello": {"https://x.com/": {"frequency": 1, "positions": [0]}}}
        filepath = str(tmp_path / "index.json")
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(expected, fh)

        loaded = load_index(filepath)
        assert loaded == expected

    def test_load_missing_file(self):
        """load_index on a non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_index("/tmp/nonexistent_path/missing.json")

    def test_round_trip_fidelity(self, tmp_path):
        """save_index followed by load_index must return a dict equal to
        the original, including _meta."""
        pages = [
            {"url": "https://example.com/a", "text": "round trip test"},
            {"url": "https://example.com/b", "text": "trip around the world"},
        ]
        original = build_index(pages)
        filepath = str(tmp_path / "rt.json")

        save_index(original, filepath)
        restored = load_index(filepath)

        assert restored == original
