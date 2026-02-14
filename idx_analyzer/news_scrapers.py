"""
Local Indonesian news scrapers for stock sentiment analysis.

Fetches news headlines from Indonesian financial news sites to supplement
Yahoo Finance's limited coverage of IDX stocks.

Supported sources:
    - Investor.id (investor.id/search/TICKER)
    - Kontan (kontan.co.id/tag/TICKER or search)
    - Detik Finance (detik.com/search with siteid=3)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Indonesian month name mapping
_BULAN = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}

# Shared HTTP headers
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Throttle between requests to be respectful
_REQUEST_DELAY = 0.8


@dataclass
class FetchedArticle:
    """A news article fetched from a local source (pre-sentiment)."""

    title: str
    publisher: str
    link: str
    published: datetime
    related_tickers: list[str] = field(default_factory=list)
    source: str = ""


def normalize_ticker(ticker: str) -> str:
    """Strip .JK suffix and IDX: prefix for local site search queries."""
    t = ticker.upper().replace(".JK", "")
    if t.startswith("IDX:"):
        t = t[4:]
    return t


def parse_indonesian_date(text: str) -> Optional[datetime]:
    """
    Parse common Indonesian date formats.

    Handles:
        - "13 Feb 2026" / "13 Februari 2026"
        - "Kamis, 13 Feb 2026 10:48 WIB"
        - "13/02/2026"
        - Relative: "12 jam yang lalu", "3 menit yang lalu"
    """
    text = text.strip()

    # Relative time: "N jam/menit/hari yang lalu"
    rel = re.match(r"(\d+)\s+(jam|menit|hari|detik)", text, re.IGNORECASE)
    if rel:
        amount = int(rel.group(1))
        unit = rel.group(2).lower()
        now = datetime.now()
        if unit == "menit":
            return now.replace(
                minute=max(0, now.minute - amount), second=0, microsecond=0
            )
        elif unit == "jam":
            return now.replace(
                hour=max(0, now.hour - amount), minute=0, second=0, microsecond=0
            )
        elif unit == "hari":
            from datetime import timedelta

            return now - timedelta(days=amount)
        return now

    # DD/MM/YYYY
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass

    # "DD Month YYYY" or "Hari, DD Month YYYY HH:MM WIB"
    # Strip day-of-week prefix if present
    text_clean = re.sub(r"^[A-Za-z]+,\s*", "", text)
    for bulan_name, bulan_num in _BULAN.items():
        # Match full or abbreviated (first 3 chars)
        pattern = rf"(\d{{1,2}})\s+{bulan_name[:3]}\w*\s+(\d{{4}})"
        m = re.search(pattern, text_clean, re.IGNORECASE)
        if m:
            try:
                day = int(m.group(1))
                year = int(m.group(2))
                return datetime(year, bulan_num, day)
            except ValueError:
                pass

    # English abbreviated months as fallback
    try:
        # Try "DD Mon YYYY" / "Mon DD, YYYY"
        for fmt in ("%d %b %Y", "%b %d, %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text_clean.split("|")[0].strip(), fmt)
            except ValueError:
                continue
    except Exception:
        pass

    return None


class NewsProvider:
    """Base class for local news providers."""

    name: str = "unknown"

    def fetch(
        self, ticker: str, max_results: int = 10, period: str = "7d"
    ) -> list[FetchedArticle]:
        raise NotImplementedError


class InvestorIdProvider(NewsProvider):
    """
    Scrapes news from investor.id search.

    URL: https://investor.id/search/TICKER
    Returns search results with article title, date, and category.
    """

    name = "Investor.id"

    def fetch(
        self, ticker: str, max_results: int = 10, period: str = "7d"
    ) -> list[FetchedArticle]:
        import requests

        query = normalize_ticker(ticker)
        url = f"https://investor.id/search/{query}"
        articles: list[FetchedArticle] = []

        try:
            logger.info(f"[{self.name}] Fetching: {url}")
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(_REQUEST_DELAY)

            soup = BeautifulSoup(resp.text, "html.parser")

            # investor.id search results are in card-like divs
            # Look for article links with titles
            for item in soup.select("article, .card, .search-result, .news-item"):
                link_tag = item.find("a", href=True)
                if not link_tag:
                    continue

                title = link_tag.get_text(strip=True)
                href = link_tag["href"]
                if not title or len(title) < 10:
                    continue

                # Make absolute URL
                if href.startswith("/"):
                    href = f"https://investor.id{href}"

                # Try to find date text
                date_el = item.find(
                    string=re.compile(
                        r"\d+\s+(jam|menit|hari)|"
                        r"\d{1,2}\s+(Jan|Feb|Mar|Apr|Mei|Jun|Jul|Agu|Sep|Okt|Nov|Des)",
                        re.IGNORECASE,
                    )
                )
                published = (
                    parse_indonesian_date(date_el.strip()) if date_el else None
                ) or datetime.now()

                articles.append(
                    FetchedArticle(
                        title=title,
                        publisher=self.name,
                        link=href,
                        published=published,
                        related_tickers=[query],
                        source=self.name,
                    )
                )

                if len(articles) >= max_results:
                    break

            # Fallback: if structured selectors didn't work, try generic link scanning
            if not articles:
                articles = self._fallback_parse(soup, query, max_results)

            logger.info(f"[{self.name}] Found {len(articles)} articles for {query}")

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to fetch news for {query}: {e}")

        return articles

    def _fallback_parse(
        self, soup: BeautifulSoup, ticker: str, max_results: int
    ) -> list[FetchedArticle]:
        """Fallback: scan all links for ticker-relevant headlines."""
        articles: list[FetchedArticle] = []
        seen_links: set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get_text(strip=True)

            # Filter: must have meaningful title, link to an article path
            if (
                not title
                or len(title) < 15
                or href in seen_links
                or not re.search(r"investor\.id/.+/.+", href)
            ):
                continue

            # Skip navigation, category, and tag links
            if re.match(
                r"https?://investor\.id/(search|tag|premium|multimedia)/?$",
                href,
            ):
                continue

            seen_links.add(href)

            # Find nearby date text
            parent = a_tag.parent
            date_text = ""
            if parent:
                date_text = parent.get_text()
            published = parse_indonesian_date(date_text) or datetime.now()

            articles.append(
                FetchedArticle(
                    title=title,
                    publisher=self.name,
                    link=href
                    if href.startswith("http")
                    else f"https://investor.id{href}",
                    published=published,
                    related_tickers=[ticker],
                    source=self.name,
                )
            )

            if len(articles) >= max_results:
                break

        return articles


class DetikFinanceProvider(NewsProvider):
    """
    Scrapes news from Detik Finance search.

    URL: https://www.detik.com/search/searchall?query=TICKER&siteid=3
    siteid=3 filters to detikFinance only.
    """

    name = "Detik Finance"

    def fetch(
        self, ticker: str, max_results: int = 10, period: str = "7d"
    ) -> list[FetchedArticle]:
        import requests

        query = normalize_ticker(ticker)
        url = (
            f"https://www.detik.com/search/searchall"
            f"?query={query}&siteid=3&result_type=latest"
        )
        articles: list[FetchedArticle] = []

        try:
            logger.info(f"[{self.name}] Fetching: {url}")
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(_REQUEST_DELAY)

            soup = BeautifulSoup(resp.text, "html.parser")

            # Detik search results: each result is in an article/list__news element
            for item in soup.select("article, .list-content__item, .media__text"):
                link_tag = item.find("a", href=True)
                if not link_tag:
                    continue

                title = link_tag.get_text(strip=True)
                href = link_tag["href"]
                if not title or len(title) < 10:
                    continue

                # Find date - Detik uses <span class="date"> or similar
                date_el = item.find("span", class_=re.compile(r"date|time", re.I))
                date_text = date_el.get_text(strip=True) if date_el else ""
                if not date_text:
                    # Try finding any text with date pattern
                    date_match = item.find(
                        string=re.compile(
                            r"(Senin|Selasa|Rabu|Kamis|Jumat|Sabtu|Minggu),\s+\d",
                            re.IGNORECASE,
                        )
                    )
                    if date_match:
                        date_text = date_match.strip()

                published = parse_indonesian_date(date_text) if date_text else None
                if not published:
                    published = datetime.now()

                articles.append(
                    FetchedArticle(
                        title=title,
                        publisher=self.name,
                        link=href,
                        published=published,
                        related_tickers=[query],
                        source=self.name,
                    )
                )

                if len(articles) >= max_results:
                    break

            # Fallback: scan links with finance.detik.com or news.detik.com URLs
            if not articles:
                articles = self._fallback_parse(soup, query, max_results)

            logger.info(f"[{self.name}] Found {len(articles)} articles for {query}")

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to fetch news for {query}: {e}")

        return articles

    def _fallback_parse(
        self, soup: BeautifulSoup, ticker: str, max_results: int
    ) -> list[FetchedArticle]:
        """Fallback: scan links matching detik article URL patterns."""
        articles: list[FetchedArticle] = []
        seen: set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get_text(strip=True)

            # Must be a detik article link (contains /d-NNNNN/)
            if (
                not title
                or len(title) < 15
                or href in seen
                or not re.search(r"/d-\d+/", href)
            ):
                continue

            seen.add(href)

            # Look for nearby date
            parent = a_tag.parent
            date_text = ""
            if parent:
                sibling = parent.find(
                    string=re.compile(r"\d{1,2}\s+\w+\s+\d{4}", re.IGNORECASE)
                )
                if sibling:
                    date_text = sibling.strip()

            published = parse_indonesian_date(date_text) or datetime.now()

            articles.append(
                FetchedArticle(
                    title=title,
                    publisher=self.name,
                    link=href,
                    published=published,
                    related_tickers=[ticker],
                    source=self.name,
                )
            )

            if len(articles) >= max_results:
                break

        return articles


class KontanProvider(NewsProvider):
    """
    Scrapes news from Kontan tag pages.

    URL: https://www.kontan.co.id/tag/TICKER
    Tag pages tend to be server-rendered and have good stock-specific content.
    Falls back to search if tag page yields no results.
    """

    name = "Kontan"

    def fetch(
        self, ticker: str, max_results: int = 10, period: str = "7d"
    ) -> list[FetchedArticle]:
        import requests

        query = normalize_ticker(ticker)

        # Try tag page first (usually better structured), then search
        urls = [
            f"https://www.kontan.co.id/tag/{query.lower()}",
            f"https://investasi.kontan.co.id/search/?search={query}",
        ]

        articles: list[FetchedArticle] = []

        for url in urls:
            if len(articles) >= max_results:
                break

            try:
                logger.info(f"[{self.name}] Fetching: {url}")
                resp = requests.get(url, headers=_HEADERS, timeout=15)
                time.sleep(_REQUEST_DELAY)

                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                page_articles = self._parse_kontan(soup, query, max_results)
                articles.extend(page_articles)

            except Exception as e:
                logger.warning(f"[{self.name}] Failed to fetch {url}: {e}")
                continue

        # Dedupe by link
        seen: set[str] = set()
        unique: list[FetchedArticle] = []
        for a in articles:
            if a.link not in seen:
                seen.add(a.link)
                unique.append(a)

        logger.info(f"[{self.name}] Found {len(unique)} articles for {query}")
        return unique[:max_results]

    def _parse_kontan(
        self, soup: BeautifulSoup, ticker: str, max_results: int
    ) -> list[FetchedArticle]:
        """Parse Kontan article listings."""
        articles: list[FetchedArticle] = []
        seen: set[str] = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get_text(strip=True)

            # Must be a kontan article link (contains /news/)
            if not title or len(title) < 15 or href in seen or "/news/" not in href:
                continue

            seen.add(href)

            # Make absolute
            if href.startswith("/"):
                href = f"https://www.kontan.co.id{href}"

            # Look for date near the link
            parent = a_tag.parent
            date_text = ""
            if parent:
                text = parent.get_text()
                # Look for "DD Mon YYYY | HH:MM WIB" or relative dates
                date_match = re.search(
                    r"(\d{1,2}\s+\w+\s+\d{4}|\d+\s+(jam|menit|hari)\s+yang\s+lalu)",
                    text,
                    re.IGNORECASE,
                )
                if date_match:
                    date_text = date_match.group(0)

            published = parse_indonesian_date(date_text) or datetime.now()

            articles.append(
                FetchedArticle(
                    title=title,
                    publisher=self.name,
                    link=href,
                    published=published,
                    related_tickers=[ticker],
                    source=self.name,
                )
            )

            if len(articles) >= max_results:
                break

        return articles


# Default provider instances
_DEFAULT_PROVIDERS: list[NewsProvider] = [
    InvestorIdProvider(),
    KontanProvider(),
    DetikFinanceProvider(),
]


def fetch_local_news(
    ticker: str,
    max_results: int = 20,
    providers: Optional[list[NewsProvider]] = None,
) -> list[FetchedArticle]:
    """
    Fetch news from all local Indonesian news providers.

    Args:
        ticker: Stock ticker (e.g., "BBCA.JK" or "BBCA")
        max_results: Maximum total articles to return
        providers: Custom provider list (default: all providers)

    Returns:
        List of FetchedArticle sorted by published date (newest first),
        deduplicated by link.
    """
    if providers is None:
        providers = _DEFAULT_PROVIDERS

    all_articles: list[FetchedArticle] = []
    per_provider = max(5, max_results // len(providers))

    for provider in providers:
        try:
            articles = provider.fetch(ticker, max_results=per_provider)
            all_articles.extend(articles)
        except Exception as e:
            logger.warning(f"Provider {provider.name} failed: {e}")
            continue

    # Dedupe by link
    seen: set[str] = set()
    unique: list[FetchedArticle] = []
    for article in all_articles:
        if article.link not in seen:
            seen.add(article.link)
            unique.append(article)

    # Sort by published date (newest first)
    unique.sort(key=lambda a: a.published, reverse=True)

    return unique[:max_results]
