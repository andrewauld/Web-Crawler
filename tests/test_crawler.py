"""
test_crawler.py — Unit tests for the BFS web crawler.

All HTTP requests are mocked — no real network calls are made.
Each test targets a single behaviour described in the coursework spec.
"""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest
import requests

# Ensure the src package is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crawler import crawl, normalise_url, is_same_domain, extract_text, extract_links
from bs4 import BeautifulSoup


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_response(html: str, status_code: int = 200) -> MagicMock:
    """Create a mock ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = html
    return resp


def _simple_page(body: str, links: list[str] | None = None) -> str:
    """Return minimal HTML with *body* text and optional <a> links."""
    link_tags = ""
    if links:
        link_tags = "".join(f'<a href="{url}">link</a>' for url in links)
    return f"<html><body>{body}{link_tags}</body></html>"


# ── Tests ────────────────────────────────────────────────────────────────────

@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_successful_crawl_returns_two_pages(mock_get, mock_sleep):
    """Mock a simple two-page site; assert both pages are returned with
    correct URLs and lowercased text."""

    page1_html = _simple_page(
        "Hello World",
        links=["https://quotes.toscrape.com/page/2"],
    )
    page2_html = _simple_page("Second Page")

    mock_get.side_effect = [
        _make_response(page1_html),
        _make_response(page2_html),
    ]

    result = crawl("https://quotes.toscrape.com/")

    assert len(result) == 2
    assert result[0]["url"] == "https://quotes.toscrape.com/"
    assert "hello world" in result[0]["text"]
    assert result[1]["url"] == "https://quotes.toscrape.com/page/2"
    assert "second page" in result[1]["text"]


class TestUrlNormalisation:
    """Assert that URLs with trailing slashes, fragments, or query strings
    are normalised/deduplicated correctly."""

    def test_strips_trailing_slash(self):
        assert normalise_url("https://quotes.toscrape.com/page/1/") == \
               "https://quotes.toscrape.com/page/1"

    def test_strips_fragment(self):
        assert normalise_url("https://quotes.toscrape.com/page/1#top") == \
               "https://quotes.toscrape.com/page/1"

    def test_preserves_query_string(self):
        """Query strings can point to different content, so they are kept."""
        url = "https://quotes.toscrape.com/search?q=love"
        assert normalise_url(url) == url

    def test_root_stays_as_slash(self):
        assert normalise_url("https://quotes.toscrape.com/") == \
               "https://quotes.toscrape.com/"

    @patch("crawler.time.sleep", return_value=None)
    @patch("crawler.requests.get")
    def test_duplicate_urls_normalised_in_crawl(self, mock_get, mock_sleep):
        """A page linking to the same URL in different forms should only
        cause one additional crawl."""
        html = _simple_page("Home", links=[
            "https://quotes.toscrape.com/page/2/",     # trailing slash
            "https://quotes.toscrape.com/page/2#top",   # fragment
            "https://quotes.toscrape.com/page/2",       # canonical
        ])
        page2_html = _simple_page("Page Two")

        mock_get.side_effect = [
            _make_response(html),
            _make_response(page2_html),
        ]

        result = crawl("https://quotes.toscrape.com/")
        assert len(result) == 2          # only 2 pages, not 4
        assert mock_get.call_count == 2  # only 2 HTTP requests made


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_skips_external_links(mock_get, mock_sleep):
    """Assert that external links are not followed."""
    html = _simple_page("Home", links=[
        "https://other.com/page",
        "https://evil.example.org/",
        "https://quotes.toscrape.com/page/2",
    ])
    page2_html = _simple_page("Internal Page")

    mock_get.side_effect = [
        _make_response(html),
        _make_response(page2_html),
    ]

    result = crawl("https://quotes.toscrape.com/")

    # Only the seed + one internal link should be crawled
    assert len(result) == 2
    urls = [p["url"] for p in result]
    assert "https://other.com/page" not in urls
    assert "https://evil.example.org" not in urls


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_does_not_crawl_same_url_twice(mock_get, mock_sleep):
    """Assert that a URL is not crawled twice even if linked from multiple pages."""
    # Page A links to B and C; page B also links back to A and to C
    page_a = _simple_page("Page A", links=[
        "https://quotes.toscrape.com/b",
        "https://quotes.toscrape.com/c",
    ])
    page_b = _simple_page("Page B", links=[
        "https://quotes.toscrape.com/",    # back-link to A
        "https://quotes.toscrape.com/c",   # duplicate link to C
    ])
    page_c = _simple_page("Page C")

    mock_get.side_effect = [
        _make_response(page_a),
        _make_response(page_b),
        _make_response(page_c),
    ]

    result = crawl("https://quotes.toscrape.com/")

    assert len(result) == 3
    assert mock_get.call_count == 3  # each URL fetched exactly once


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_skips_page_on_request_exception(mock_get, mock_sleep):
    """Mock a requests.RequestException; assert the crawler skips and continues."""
    page1_html = _simple_page("Home", links=[
        "https://quotes.toscrape.com/bad",
        "https://quotes.toscrape.com/good",
    ])
    good_html = _simple_page("Good Page")

    mock_get.side_effect = [
        _make_response(page1_html),                         # seed OK
        requests.RequestException("Connection refused"),     # /bad fails
        _make_response(good_html),                           # /good OK
    ]

    result = crawl("https://quotes.toscrape.com/")

    # The failed page should be skipped, not crash
    assert len(result) == 2
    urls = [p["url"] for p in result]
    assert "https://quotes.toscrape.com/bad" not in urls
    assert "https://quotes.toscrape.com/good" in urls


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_skips_page_on_non_200_status(mock_get, mock_sleep):
    """Mock a 404 response; assert the page is skipped gracefully."""
    page1_html = _simple_page("Home", links=[
        "https://quotes.toscrape.com/missing",
        "https://quotes.toscrape.com/ok",
    ])
    ok_html = _simple_page("OK Page")

    mock_get.side_effect = [
        _make_response(page1_html),                     # seed
        _make_response("Not Found", status_code=404),   # /missing
        _make_response(ok_html),                        # /ok
    ]

    result = crawl("https://quotes.toscrape.com/")

    assert len(result) == 2
    urls = [p["url"] for p in result]
    assert "https://quotes.toscrape.com/missing" not in urls


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_politeness_delay_between_requests(mock_get, mock_sleep):
    """Assert that time.sleep is called with a value >= 6 between requests."""
    page1 = _simple_page("Home", links=["https://quotes.toscrape.com/page2"])
    page2 = _simple_page("Two")

    mock_get.side_effect = [
        _make_response(page1),
        _make_response(page2),
    ]

    crawl("https://quotes.toscrape.com/")

    # sleep should be called once (between the 1st and 2nd request)
    assert mock_sleep.call_count == 1
    delay = mock_sleep.call_args[0][0]
    assert delay >= 6, f"Politeness delay was {delay}s, expected >= 6s"


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_text_is_lowercased(mock_get, mock_sleep):
    """Assert that returned text is fully lowercased regardless of source casing."""
    html = _simple_page("HELLO World FoO BAR")

    mock_get.side_effect = [_make_response(html)]

    result = crawl("https://quotes.toscrape.com/")

    assert result[0]["text"] == result[0]["text"].lower()
    assert "hello world foo bar" in result[0]["text"]


@patch("crawler.time.sleep", return_value=None)
@patch("crawler.requests.get")
def test_empty_page_does_not_crash(mock_get, mock_sleep):
    """Mock a page with no visible text; assert it is handled gracefully."""
    html = "<html><body></body></html>"

    mock_get.side_effect = [_make_response(html)]

    result = crawl("https://quotes.toscrape.com/")

    assert len(result) == 1
    assert result[0]["url"] == "https://quotes.toscrape.com/"
    assert result[0]["text"].strip() == ""


# ── Unit tests for helper functions ──────────────────────────────────────────

class TestIsSameDomain:
    def test_same_domain(self):
        assert is_same_domain("https://quotes.toscrape.com/page/1") is True

    def test_different_domain(self):
        assert is_same_domain("https://other.com/page") is False

    def test_subdomain_rejected(self):
        assert is_same_domain("https://sub.quotes.toscrape.com/") is False


class TestExtractText:
    def test_strips_tags(self):
        soup = BeautifulSoup("<p>Hello <b>World</b></p>", "html.parser")
        assert extract_text(soup) == "hello world"

    def test_collapses_whitespace(self):
        soup = BeautifulSoup("<p>  lots   of   space  </p>", "html.parser")
        assert "  " not in extract_text(soup)


class TestExtractLinks:
    def test_resolves_relative_links(self):
        soup = BeautifulSoup('<a href="/page/2">next</a>', "html.parser")
        links = extract_links(soup, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page/2" in links

    def test_filters_external_links(self):
        soup = BeautifulSoup('<a href="https://evil.com/x">x</a>', "html.parser")
        links = extract_links(soup, "https://quotes.toscrape.com/")
        assert len(links) == 0
