"""
test_search.py — Pytest test suite for the search component

All tests use fixture-based mock indices (small, hand-crafted dicts with
known, calculable expected values).  No disk I/O or network calls.
"""

import math
import os
import sys

import pytest

# Ensure the src package is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search import compute_tfidf, print_word, search_and, search_or, suggest


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_index():
    """Small index with three pages and three words for calculable TF-IDF."""
    return {
        "_meta": {
            "http://example.com/a": {"total_tokens": 10},
            "http://example.com/b": {"total_tokens": 20},
            "http://example.com/c": {"total_tokens": 5},
        },
        "world": {
            "http://example.com/a": {"frequency": 2, "positions": [0, 5]},
            "http://example.com/b": {"frequency": 1, "positions": [3]},
        },
        "hello": {
            "http://example.com/a": {"frequency": 1, "positions": [1]},
            "http://example.com/c": {"frequency": 2, "positions": [0, 4]},
        },
        "nonsensical": {
            "http://example.com/b": {"frequency": 1, "positions": [7]},
        },
    }


@pytest.fixture
def zero_tokens_index():
    """Index where one page has total_tokens == 0 (edge case)."""
    return {
        "_meta": {
            "http://example.com/x": {"total_tokens": 0},
        },
        "test": {
            "http://example.com/x": {"frequency": 1, "positions": [0]},
        },
    }


@pytest.fixture
def suggest_index():
    """Index tailored for suggestion/stemming tests with multiple related words."""
    return {
        "_meta": {
            "http://example.com/a": {"total_tokens": 10},
        },
        "running": {
            "http://example.com/a": {"frequency": 1, "positions": [0]},
        },
        "runs": {
            "http://example.com/a": {"frequency": 1, "positions": [1]},
        },
        "runner": {
            "http://example.com/a": {"frequency": 1, "positions": [2]},
        },
        "unrelated": {
            "http://example.com/a": {"frequency": 1, "positions": [3]},
        },
    }


# ── print_word tests ────────────────────────────────────────────────────────


class TestPrintWord:
    """Tests for the print_word function."""

    def test_word_exists(self, sample_index):
        """Should return the correct postings dict for a known word."""
        result = print_word(sample_index, "world")
        assert result == {
            "http://example.com/a": {"frequency": 2, "positions": [0, 5]},
            "http://example.com/b": {"frequency": 1, "positions": [3]},
        }

    def test_word_not_found(self, sample_index):
        """Should return None when the word is not in the index."""
        result = print_word(sample_index, "banana")
        assert result is None

    def test_case_insensitive(self, sample_index):
        """Should lowercase the query so 'World' and 'world' match the same entry."""
        assert print_word(sample_index, "World") == print_word(sample_index, "world")

    def test_meta_not_returned(self, sample_index):
        """Should return None for '_meta' — it is not a vocabulary word."""
        assert print_word(sample_index, "_meta") is None


# ── compute_tfidf tests ─────────────────────────────────────────────────────


class TestComputeTfidf:
    """Tests for the compute_tfidf function."""

    def test_known_value(self, sample_index):
        """Hand-calculated TF-IDF for 'world' on page /a.

        TF  = 2 / 10 = 0.2
        IDF = ln(3 / 2) ≈ 0.405465
        TF-IDF ≈ 0.08109
        """
        expected_tf = 2 / 10
        expected_idf = math.log(3 / 2)
        expected = expected_tf * expected_idf

        result = compute_tfidf(sample_index, "world", "http://example.com/a")
        assert result == pytest.approx(expected)

    def test_word_not_in_index(self, sample_index):
        """Should return 0.0 when the word does not exist in the index."""
        assert compute_tfidf(sample_index, "banana", "http://example.com/a") == 0.0

    def test_url_not_in_postings(self, sample_index):
        """Should return 0.0 when the URL is not in the word's postings."""
        # "nonsensical" only appears on page /b
        assert compute_tfidf(sample_index, "nonsensical", "http://example.com/a") == 0.0

    def test_zero_total_tokens(self, zero_tokens_index):
        """Should return 0.0 without raising when total_tokens is 0."""
        result = compute_tfidf(zero_tokens_index, "test", "http://example.com/x")
        assert result == 0.0

    def test_empty_index(self):
        """Should return 0.0 when total_pages is 0."""
        empty_index = {"_meta": {}}
        assert compute_tfidf(empty_index, "any", "http://example.com/a") == 0.0


# ── search_and tests ────────────────────────────────────────────────────────


class TestSearchAnd:
    """Tests for the search_and function (AND / intersection logic)."""

    def test_single_term(self, sample_index):
        """Should return all URLs containing the single term, ranked by TF-IDF."""
        results = search_and(sample_index, ["world"])
        urls = [url for url, _ in results]
        # "world" appears on pages /a and /b
        assert set(urls) == {"http://example.com/a", "http://example.com/b"}

    def test_multi_term_intersection(self, sample_index):
        """Only page /a contains both 'world' and 'hello'."""
        results = search_and(sample_index, ["world", "hello"])
        urls = [url for url, _ in results]
        assert urls == ["http://example.com/a"]

    def test_no_intersection(self, sample_index):
        """'hello' and 'nonsensical' share no pages — should return []."""
        results = search_and(sample_index, ["hello", "nonsensical"])
        assert results == []

    def test_term_not_in_index(self, sample_index):
        """A term missing from the index makes the intersection empty."""
        results = search_and(sample_index, ["world", "banana"])
        assert results == []

    def test_ranking_order(self, sample_index):
        """Results should be sorted by score descending."""
        results = search_and(sample_index, ["world"])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self, sample_index):
        """An empty query list should return [] without raising."""
        assert search_and(sample_index, []) == []

    def test_aggregate_scores(self, sample_index):
        """For multi-term AND, the score should be the sum of per-term TF-IDFs."""
        results = search_and(sample_index, ["world", "hello"])
        assert len(results) == 1
        url, score = results[0]
        expected = (
            compute_tfidf(sample_index, "world", url)
            + compute_tfidf(sample_index, "hello", url)
        )
        assert score == pytest.approx(expected)


# ── search_or tests ─────────────────────────────────────────────────────────


class TestSearchOr:
    """Tests for the search_or function (OR / union logic)."""

    def test_single_term(self, sample_index):
        """For a single term, OR should return the same URLs as AND."""
        and_results = search_and(sample_index, ["world"])
        or_results = search_or(sample_index, ["world"])
        assert [url for url, _ in and_results] == [url for url, _ in or_results]

    def test_multi_term_union(self, sample_index):
        """Union of 'hello' and 'nonsensical' should include pages /a, /b, /c."""
        results = search_or(sample_index, ["hello", "nonsensical"])
        urls = {url for url, _ in results}
        assert urls == {
            "http://example.com/a",
            "http://example.com/b",
            "http://example.com/c",
        }

    def test_ranking_order(self, sample_index):
        """Results should be sorted by score descending."""
        results = search_or(sample_index, ["world", "hello"])
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_terms_missing(self, sample_index):
        """Should return [] when no query term exists in the index."""
        assert search_or(sample_index, ["banana", "pineapple"]) == []

    def test_empty_query(self, sample_index):
        """An empty query list should return [] without raising."""
        assert search_or(sample_index, []) == []

    def test_or_includes_partial_matches(self, sample_index):
        """Pages matching only some terms should still appear (with lower scores)."""
        results = search_or(sample_index, ["world", "hello"])
        urls = {url for url, _ in results}
        # All three pages should appear: /a has both, /b has "world", /c has "hello"
        assert urls == {
            "http://example.com/a",
            "http://example.com/b",
            "http://example.com/c",
        }


# ── suggest tests ────────────────────────────────────────────────────────────


class TestSuggest:
    """Tests for the suggest function (stemming-based suggestions)."""

    def test_finds_related_word(self, sample_index):
        """'nonsense' should stem to the same root as 'nonsensical'."""
        result = suggest(sample_index, "nonsense")
        assert "nonsensical" in result

    def test_excludes_query_word(self, sample_index):
        """The query word itself must not appear in the suggestion list."""
        result = suggest(sample_index, "nonsensical")
        assert "nonsensical" not in result

    def test_no_related_words(self, sample_index):
        """Should return [] when no vocabulary word shares a stem."""
        result = suggest(sample_index, "banana")
        assert result == []

    def test_case_insensitive(self, sample_index):
        """'Nonsense' and 'nonsense' should produce the same suggestions."""
        assert suggest(sample_index, "Nonsense") == suggest(sample_index, "nonsense")

    def test_alphabetical_order(self, suggest_index):
        """When multiple suggestions exist, they must be sorted alphabetically."""
        result = suggest(suggest_index, "run")
        # "run" stems to "run"; "running", "runs", "runner" all stem to "run"
        assert result == sorted(result)
        assert len(result) >= 2  # at least some of {running, runs, runner}

    def test_query_word_in_index_excluded(self, suggest_index):
        """If the query word is in the index, it should be excluded from results."""
        result = suggest(suggest_index, "running")
        assert "running" not in result
        # But other related words should still appear
        assert len(result) >= 1
