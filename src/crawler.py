"""
crawler.py — BFS web crawler for quotes.toscrape.com

Crawls the entire target site using breadth-first search, respecting a
politeness window of ≥6 seconds between requests. Returns a list of
page dictionaries containing the URL and lowercased visible text.
"""

import re
import time
import logging
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

SEED_URL = "https://quotes.toscrape.com/"
ALLOWED_DOMAIN = "quotes.toscrape.com"
POLITENESS_DELAY = 6          # seconds between successive requests
REQUEST_TIMEOUT = 10          # seconds before a request times out


def normalise_url(url: str) -> str:
    """Normalise a URL for deduplication.

    Strips:
      - fragment (#section)
      - trailing slash on the path
      - default ports (80/443)

    This prevents the same page from being crawled under different
    surface-level URL variants.
    """
    parsed = urlparse(url)

    # Remove fragment
    # Remove default ports (netloc might include :80 / :443)
    netloc = parsed.netloc
    if parsed.scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif parsed.scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Strip trailing slash from path (but keep "/" for root)
    path = parsed.path.rstrip("/") or "/"

    return urlunparse((parsed.scheme, netloc, path, "", parsed.query, ""))


def is_same_domain(url: str) -> bool:
    """Return True if *url* belongs to the allowed domain."""
    return urlparse(url).netloc == ALLOWED_DOMAIN


def extract_text(soup: BeautifulSoup) -> str:
    """Extract visible text from a parsed page.

    Uses BeautifulSoup's get_text() with a space separator, collapses
    excessive whitespace, and lowercases the result (case-insensitive
    indexing requirement).
    """
    raw = soup.get_text(separator=" ", strip=True)
    # Collapse any runs of whitespace into a single space
    collapsed = re.sub(r"\s+", " ", raw).strip()
    return collapsed.lower()


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Return a list of absolute, normalised, same-domain URLs found on the page."""
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        absolute = urljoin(base_url, href)          # resolve relative URLs
        normalised = normalise_url(absolute)
        if is_same_domain(normalised):
            links.append(normalised)
    return links


def crawl(seed_url: str = SEED_URL) -> list[dict]:
    """Crawl the target site starting from *seed_url* using BFS.

    Returns a list of dicts, each with keys ``url`` and ``text``::

        [{"url": "https://quotes.toscrape.com/", "text": "..."}, ...]

    The crawler:
      - uses a FIFO queue (deque) for breadth-first traversal
      - tracks visited URLs to avoid re-crawling
      - sleeps ≥6 s between requests (politeness requirement)
      - skips pages that error or return non-200 status codes
    """
    visited: set[str] = set()
    queue: deque[str] = deque()
    pages: list[dict] = []

    start = normalise_url(seed_url)
    queue.append(start)
    visited.add(start)

    first_request = True  # no delay before the very first request

    while queue:
        url = queue.popleft()

        # ── Politeness: wait between successive requests ─────────────
        if not first_request:
            time.sleep(POLITENESS_DELAY)
        first_request = False

        # ── Fetch the page ───────────────────────────────────────────
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            continue                                   # skip and carry on

        if response.status_code != 200:
            logger.warning(
                "Non-200 status (%d) for %s — skipping",
                response.status_code,
                url,
            )
            continue

        # ── Parse and store ──────────────────────────────────────────
        soup = BeautifulSoup(response.text, "html.parser")
        text = extract_text(soup)
        pages.append({"url": url, "text": text})
        print(f"Crawling page {len(pages)}: {url}")

        # ── Discover new links ───────────────────────────────────────
        for link in extract_links(soup, url):
            if link not in visited:
                visited.add(link)
                queue.append(link)

    logger.info("Crawl complete: %d pages collected.", len(pages))
    return pages
