"""
Sentiment Analysis Module for IDX Stock Analyzer

Uses Yahoo Finance news data and FinBERT model for financial sentiment analysis.
No API registration required - runs completely locally after model download.

Requirements:
    pip install transformers torch

Optional (lighter alternative):
    pip install vaderSentiment  # Lightweight, no ML model download
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import logging

logger = logging.getLogger(__name__)


def _load_indonesian_lexicon() -> dict[str, set[str]]:
    """
    Load Indonesian financial sentiment lexicon from JSON file.

    Returns:
        Dictionary with keys: 'positive', 'negative', 'negation', 'uncertainty'
        Each containing a set of terms for matching.
    """
    lexicon_path = Path(__file__).parent / "indonesian_sentiment_lexicon.json"

    try:
        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten positive terms from all subcategories
        positive_terms = set()
        for category_data in data.get("positive", {}).values():
            if isinstance(category_data, dict) and "terms" in category_data:
                positive_terms.update(category_data["terms"])

        # Flatten negative terms from all subcategories
        negative_terms = set()
        for category_data in data.get("negative", {}).values():
            if isinstance(category_data, dict) and "terms" in category_data:
                negative_terms.update(category_data["terms"])

        # Get modifier terms
        negation_terms = set(
            data.get("modifiers", {}).get("negation", {}).get("terms", [])
        )
        uncertainty_terms = set(
            data.get("modifiers", {}).get("uncertainty", {}).get("terms", [])
        )

        return {
            "positive": positive_terms,
            "negative": negative_terms,
            "negation": negation_terms,
            "uncertainty": uncertainty_terms,
        }

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not load Indonesian lexicon from {lexicon_path}: {e}")
        logger.warning("Using empty lexicon - hybrid sentiment analysis disabled")
        return {
            "positive": set(),
            "negative": set(),
            "negation": set(),
            "uncertainty": set(),
        }


# Load the Indonesian sentiment lexicon on module import
_INDO_LEXICON = _load_indonesian_lexicon()

# Export term sets for backward compatibility
INDONESIAN_POSITIVE_TERMS = _INDO_LEXICON["positive"]
INDONESIAN_NEGATIVE_TERMS = _INDO_LEXICON["negative"]
INDONESIAN_NEGATION_TERMS = _INDO_LEXICON["negation"]
INDONESIAN_UNCERTAINTY_TERMS = _INDO_LEXICON["uncertainty"]


# Boost multipliers for specific patterns
SENTIMENT_BOOST_STRONG = 0.35
SENTIMENT_BOOST_MEDIUM = 0.20
SENTIMENT_BOOST_WEAK = 0.10


def _normalize_text(text: str) -> str:
    """Normalize Indonesian text for matching."""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _count_term_matches(text: str, term_set: set[str]) -> tuple[int, float]:
    """
    Count matches and calculate weighted score based on term strength.
    Returns (count, weighted_score)
    """
    text_normalized = _normalize_text(text)
    count = 0
    weighted_score = 0.0
    matched_terms = set()

    # Sort terms by length (longest first) to match multi-word phrases before single words
    sorted_terms = sorted(term_set, key=len, reverse=True)

    for term in sorted_terms:
        # Skip if this term is a substring of already matched longer term
        if any(
            term in matched for matched in matched_terms if len(matched) > len(term)
        ):
            continue

        if term in text_normalized:
            count += 1
            matched_terms.add(term)
            # Stronger weighting for matches
            # Base weight: 1.0 for single word, up to 2.5 for phrases
            word_count = len(term.split())
            if word_count == 1:
                weight = 1.0
            elif word_count == 2:
                weight = 1.8
            else:
                weight = 2.5
            weighted_score += weight

    return count, weighted_score


def _detect_negation_context(text: str, term_position: int) -> bool:
    """Check if there's a negation word before a given position in text."""
    text_normalized = _normalize_text(text)
    words_before = text_normalized[:term_position].split()[-5:]  # Last 5 words
    return any(neg in words_before for neg in INDONESIAN_NEGATION_TERMS)


def apply_indonesian_sentiment_rules(
    text: str, base_label: str, base_score: float
) -> tuple[str, float, bool]:
    """
    Apply Indonesian market-specific rules to adjust FinBERT sentiment.

    This hybrid approach:
    1. Takes FinBERT's base prediction
    2. Scans for Indonesian financial terms
    3. Adjusts label/score based on term matches and context
    4. Returns enhanced sentiment

    Args:
        text: Original headline/text
        base_label: FinBERT's predicted label ('positive', 'negative', 'neutral')
        base_score: FinBERT's confidence score (0.0 to 1.0)

    Returns:
        Tuple of (adjusted_label, adjusted_score, has_indonesian_terms)
    """
    text_lower = text.lower()

    # Count term matches
    pos_count, pos_weight = _count_term_matches(text, INDONESIAN_POSITIVE_TERMS)
    neg_count, neg_weight = _count_term_matches(text, INDONESIAN_NEGATIVE_TERMS)

    # Check if text appears to be Indonesian
    is_indonesian = bool(
        re.search(
            r"[aiueo]ng|[aiueo]ny|[aiueo]r\b|yang|dengan|untuk|dari|dalam|pada|oleh|saham|harga|bank|bursa",
            text_lower,
        )
    )
    indonesian_term_ratio = (pos_count + neg_count) / max(len(text.split()), 1)
    # Detect if ANY Indonesian financial terms present - single term triggers mode
    has_indonesian_terms = (
        pos_count + neg_count > 0 or indonesian_term_ratio > 0.02 or is_indonesian
    )

    # Calculate sentiment delta based on term weights - STRONGLY AMPLIFIED
    sentiment_delta = (pos_weight - neg_weight) * 0.4  # Stronger multiplier

    # Check for uncertainty modifiers
    uncertainty_count, _ = _count_term_matches(text, INDONESIAN_UNCERTAINTY_TERMS)
    uncertainty_factor = max(0.7, 1.0 - (uncertainty_count * 0.1))

    # Determine rule-based sentiment direction - VERY LOW THRESHOLDS
    if pos_weight > neg_weight * 1.05:  # Almost any positive weight advantage
        rule_sentiment = "positive"
        rule_confidence = min(0.6 + (pos_weight * 0.15), 0.95)
    elif neg_weight > pos_weight * 1.05:  # Almost any negative weight advantage
        rule_sentiment = "negative"
        rule_confidence = min(0.6 + (neg_weight * 0.15), 0.95)
    else:
        rule_sentiment = "neutral"
        rule_confidence = 0.5

    # Hybrid decision: combine FinBERT with rule-based
    rule_strength = max(pos_weight, neg_weight)

    # Decision logic - VERY AGGRESSIVE for Indonesian context
    if has_indonesian_terms:
        # Strong Indonesian context detected
        if rule_strength >= 0.5 and rule_sentiment != base_label:
            # Clear term-based signal contradicts FinBERT - trust rules heavily
            adjusted_label = rule_sentiment
            adjusted_score = rule_confidence * uncertainty_factor
        elif rule_strength >= 0.3:  # Low threshold triggers adjustment
            # Apply sentiment delta adjustment
            adjusted_label = _adjust_label(base_label, sentiment_delta)
            adjusted_score = (
                min(max(base_score, 0.6) + abs(sentiment_delta), 0.95)
                * uncertainty_factor
            )
        else:
            # Weak term signals but Indonesian context - still adjust slightly
            adjusted_label = _adjust_label(base_label, sentiment_delta * 0.5)
            adjusted_score = max(base_score, 0.55) * uncertainty_factor
    else:
        # Weak Indonesian context - trust FinBERT more
        adjusted_label = base_label
        adjusted_score = base_score

    # Final confidence boost if strong signals align
    if adjusted_label == rule_sentiment and rule_strength >= 1.0:
        adjusted_score = min(adjusted_score * 1.15, 0.95)

    return adjusted_label, round(adjusted_score, 3), has_indonesian_terms


def _adjust_label(current_label: str, delta: float) -> str:
    """Adjust label based on sentiment delta."""
    # Very low thresholds for maximum responsiveness
    if delta > 0.08:  # Very sensitive - any positive signal flips to positive
        return "positive"
    elif delta < -0.08:  # Very sensitive - any negative signal flips to negative
        return "negative"
    elif abs(delta) < 0.03:  # Near zero - stick with current
        return current_label
    else:
        # Small delta - if currently neutral, lean toward direction
        if current_label == "neutral":
            if delta > 0:
                return "positive"
            elif delta < 0:
                return "negative"
        # If already has sentiment, keep it unless delta strongly opposite
        return current_label


@dataclass
class NewsArticle:
    """Represents a news article with sentiment"""

    title: str
    publisher: str
    link: str
    published: datetime
    sentiment_label: Literal["positive", "negative", "neutral"]
    sentiment_score: float  # 0.0 to 1.0 (confidence)
    related_tickers: list[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Complete sentiment analysis result"""

    ticker: str
    aggregate_score: float  # -1.0 to +1.0
    positive_count: int
    negative_count: int
    neutral_count: int
    total_articles: int
    articles: list[NewsArticle]
    analysis_period: str = "7d"  # Default: last 7 days
    model_used: str = "finbert-hybrid"  # finbert-hybrid, finbert, vader, or llm

    @property
    def sentiment_label(self) -> str:
        """Get overall sentiment label"""
        if self.aggregate_score > 0.2:
            return "bullish"
        elif self.aggregate_score < -0.2:
            return "bearish"
        else:
            return "neutral"

    @property
    def sentiment_emoji(self) -> str:
        """Get emoji representation"""
        if self.aggregate_score > 0.2:
            return "📈"
        elif self.aggregate_score < -0.2:
            return "📉"
        else:
            return "⚖️"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "aggregate_score": round(self.aggregate_score, 3),
            "sentiment": self.sentiment_label,
            "positive": self.positive_count,
            "negative": self.negative_count,
            "neutral": self.neutral_count,
            "total_articles": self.total_articles,
            "model": self.model_used,
            "articles": [
                {
                    "title": a.title,
                    "publisher": a.publisher,
                    "sentiment": a.sentiment_label,
                    "confidence": round(a.sentiment_score, 3),
                    "published": a.published.isoformat(),
                }
                for a in self.articles
            ],
        }


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using Yahoo Finance news and FinBERT.

    FinBERT is a pre-trained NLP model specifically for financial text.
    It classifies text into: positive, negative, or neutral sentiment.

    Alternative: Use VADER for lightweight sentiment without model download.
    Alternative: Use LLM via OpenAI-compatible API (e.g., Ollama) for
                 multilingual sentiment that understands Bahasa Indonesia.

    HYBRID MODE (default for FinBERT):
    When using FinBERT with Indonesian stocks, the analyzer automatically
    applies a rule-based layer that recognizes Indonesian financial terms
    and adjusts FinBERT's predictions accordingly. This improves accuracy
    on Bahasa Indonesia headlines significantly.

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("BBCA.JK")
        >>> print(f"Sentiment: {result.sentiment_label} ({result.aggregate_score:+.2f})")
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finBERT",
        use_vader: bool = False,
        use_llm: bool = False,
        use_hybrid: bool = True,
        llm_model: str = "deepseek-v3.1:671b-cloud",
        llm_base_url: str = "http://localhost:11434/v1",
        llm_api_key: str = "ollama",
    ):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finBERT)
            use_vader: If True, use VADER instead of FinBERT (no download, lighter)
            use_llm: If True, use LLM via OpenAI-compatible API
            use_hybrid: If True (default), enable Indonesian term enhancement for FinBERT
            llm_model: LLM model name for the API (default: deepseek-v3.1:671b-cloud)
            llm_base_url: OpenAI-compatible API base URL (default: Ollama local)
            llm_api_key: API key (default: "ollama" for local Ollama)
        """
        self.model_name = model_name
        self.use_vader = use_vader
        self.use_llm = use_llm
        self.use_hybrid = use_hybrid
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self._pipeline = None
        self._analyzer = None
        self._llm_client = None

    def _get_pipeline(self):
        """Lazy load the NLP pipeline"""
        if self.use_llm:
            return self._get_llm_client()

        if self._pipeline is not None:
            return self._pipeline

        if self.use_vader:
            # Use VADER (lightweight, no model download)
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._analyzer = SentimentIntensityAnalyzer()
                logger.info("Using VADER sentiment analyzer")
            except ImportError:
                raise ImportError(
                    "VADER not installed. Run: pip install vaderSentiment"
                )
        else:
            # Use FinBERT (better for financial text, requires download)
            try:
                from transformers import pipeline

                logger.info(f"Loading FinBERT model: {self.model_name}")
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                )
                logger.info("FinBERT model loaded successfully")
            except ImportError:
                raise ImportError(
                    "Transformers not installed. Run: pip install transformers torch"
                )

        return self._pipeline

    def _get_llm_client(self):
        """Get or create the OpenAI-compatible LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        from openai import OpenAI

        self._llm_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        logger.info(f"Using LLM sentiment: {self.llm_model} @ {self.llm_base_url}")
        return self._llm_client

    def _analyze_text_llm(self, headlines: list[str]) -> list[tuple[str, float]]:
        """
        Batch-analyze headlines using LLM via OpenAI-compatible API.

        Sends all headlines in a single prompt for efficiency.
        The LLM understands Bahasa Indonesia natively.

        Returns:
            List of (label, confidence) tuples, one per headline.
        """
        import json as json_mod

        client = self._get_llm_client()

        # Build numbered list
        numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))

        prompt = f"""You are a financial sentiment classifier for Indonesian Stock Exchange (IDX) news.

Analyze each headline and classify its sentiment for stock investors.

Headlines:
{numbered}

Respond with ONLY a JSON array. Each element must be an object with:
- "label": one of "positive", "negative", or "neutral"
- "score": confidence from 0.0 to 1.0

Rules:
- Headlines about price drops, sell-offs, losses, downgrades = negative
- Headlines about price gains, upgrades, dividends, growth = positive
- Neutral if purely informational or ambiguous
- You understand both Bahasa Indonesia and English

Example response for 2 headlines:
[{{"label":"negative","score":0.85}},{{"label":"positive","score":0.9}}]

Respond with ONLY the JSON array, no other text."""

        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON array from response (handle markdown code blocks)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            results = json_mod.loads(content)

            # Validate and normalize
            parsed: list[tuple[str, float]] = []
            for r in results:
                label = r.get("label", "neutral").lower()
                if label not in ("positive", "negative", "neutral"):
                    label = "neutral"
                score = float(r.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                parsed.append((label, score))

            # Pad if LLM returned fewer results than headlines
            while len(parsed) < len(headlines):
                parsed.append(("neutral", 0.5))

            return parsed[: len(headlines)]

        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed: {e}")
            # Fall back to neutral for all
            return [("neutral", 0.5)] * len(headlines)

    def _analyze_text(self, text: str) -> tuple[str, float, bool]:
        """
        Analyze single text and return sentiment label, score, and hybrid status.

        Returns:
            Tuple of (label, confidence_score, is_hybrid_adjusted)
        """
        # Truncate to 512 tokens (FinBERT limit)
        text = text[:512]

        if self.use_vader:
            # VADER analysis
            scores = self._analyzer.polarity_scores(text)
            compound = scores["compound"]

            # Convert to label
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            # Convert compound (-1 to 1) to confidence (0 to 1)
            confidence = abs(compound)
            return label, confidence, False
        else:
            # FinBERT analysis with optional Indonesian hybrid enhancement
            pipeline = self._get_pipeline()
            result = pipeline(text)[0]
            base_label = result["label"].lower()
            base_score = result["score"]

            # Apply hybrid Indonesian sentiment rules if enabled
            if self.use_hybrid:
                adjusted_label, adjusted_score, has_indonesian_terms = (
                    apply_indonesian_sentiment_rules(text, base_label, base_score)
                )
                return adjusted_label, adjusted_score, has_indonesian_terms
            else:
                return base_label, base_score, False

    def _fetch_yahoo_news(self, ticker: str, max_articles: int) -> list[dict]:
        """Fetch news from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        logger.info(f"Fetching Yahoo Finance news for {ticker}")
        stock = yf.Ticker(ticker)

        try:
            news_items = stock.get_news(count=max_articles, tab="news")
        except Exception as e:
            logger.warning(f"get_news() failed: {e}, trying .news property")
            news_items = stock.news[:max_articles] if stock.news else []

        return news_items or []

    def _fetch_local_news(self, ticker: str, max_articles: int) -> list[dict]:
        """
        Fetch news from local Indonesian sources and convert to Yahoo-like dicts.

        Sources: Investor.id, Kontan, Detik Finance
        """
        try:
            from .news_scrapers import fetch_local_news
        except ImportError as e:
            logger.warning(f"Local news scrapers unavailable: {e}")
            return []

        try:
            articles = fetch_local_news(ticker, max_results=max_articles)
            # Convert FetchedArticle to Yahoo-like dict format for unified processing
            return [
                {
                    "title": a.title,
                    "publisher": a.publisher,
                    "link": a.link,
                    "published": int(a.published.timestamp()),
                    "relatedTickers": a.related_tickers,
                    "_source": a.source,
                }
                for a in articles
            ]
        except Exception as e:
            logger.warning(f"Local news fetch failed: {e}")
            return []

    def analyze(
        self,
        ticker: str,
        max_articles: int = 20,
        period: str = "7d",
    ) -> SentimentResult:
        """
        Analyze sentiment for a stock ticker.

        Fetches news from Yahoo Finance and local Indonesian sources
        (Investor.id, Kontan, Detik Finance), then analyzes headline sentiment.

        Args:
            ticker: Stock ticker (e.g., "BBCA.JK", "TLKM.JK")
            max_articles: Maximum number of news articles to analyze
            period: Analysis period (not used currently, Yahoo returns recent news)

        Returns:
            SentimentResult with aggregate score and individual article analysis

        Raises:
            ImportError: If yfinance or transformers not installed
            Exception: If news fetch or analysis fails
        """
        # Initialize pipeline (will download model on first run)
        self._get_pipeline()

        # Fetch from Yahoo Finance
        yahoo_items = self._fetch_yahoo_news(ticker, max_articles)
        yahoo_count = len(yahoo_items)
        logger.info(f"Yahoo Finance returned {yahoo_count} articles")

        # Supplement with local Indonesian news sources
        local_needed = max(5, max_articles - yahoo_count)
        local_items = self._fetch_local_news(ticker, local_needed)
        logger.info(f"Local sources returned {len(local_items)} articles")

        # Merge and deduplicate
        all_items = yahoo_items + local_items
        seen_titles: set[str] = set()
        unique_items: list[dict] = []
        for item in all_items:
            title = item.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_items.append(item)

        if not unique_items:
            logger.warning(f"No news found for {ticker}")
            return SentimentResult(
                ticker=ticker,
                aggregate_score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_articles=0,
                articles=[],
                analysis_period=period,
                model_used=self._model_label,
            )

        # Analyze each article
        analyzed_articles: list[NewsArticle] = []
        pos_count = neg_count = neutral_count = 0

        if self.use_llm:
            # Batch analysis via LLM (single API call for all headlines)
            headlines = [
                item.get("title", "") for item in unique_items if item.get("title")
            ]
            items_with_titles = [item for item in unique_items if item.get("title")]

            if headlines:
                llm_results = self._analyze_text_llm(headlines)

                for item, (label, score) in zip(items_with_titles, llm_results):
                    title = item.get("title", "")
                    published_ts = item.get("published") or item.get("date", 0)
                    try:
                        published = datetime.fromtimestamp(published_ts)
                    except (ValueError, TypeError, OSError):
                        published = datetime.now()

                    source = item.get("_source", "Yahoo Finance")
                    publisher = item.get("publisher", "Unknown")
                    if source and source != "Yahoo Finance":
                        publisher = f"{publisher} ({source})"

                    article = NewsArticle(
                        title=title,
                        publisher=publisher,
                        link=item.get("link", ""),
                        published=published,
                        sentiment_label=label,  # type: ignore
                        sentiment_score=score,
                        related_tickers=item.get("relatedTickers", []),
                    )
                    analyzed_articles.append(article)
        else:
            # Per-headline analysis via FinBERT or VADER
            for item in unique_items:
                title = item.get("title", "")
                if not title:
                    continue

                try:
                    label, score, has_indonesian_terms = self._analyze_text(title)

                    # Low-confidence guard for non-English headlines
                    # Skip guard if Indonesian terms detected and hybrid mode adjusted sentiment
                    if score < 0.55 and not has_indonesian_terms:
                        label = "neutral"

                    # Parse timestamp
                    published_ts = item.get("published") or item.get("date", 0)
                    try:
                        published = datetime.fromtimestamp(published_ts)
                    except (ValueError, TypeError, OSError):
                        published = datetime.now()

                    source = item.get("_source", "Yahoo Finance")
                    publisher = item.get("publisher", "Unknown")
                    if source and source != "Yahoo Finance":
                        publisher = f"{publisher} ({source})"

                    article = NewsArticle(
                        title=title,
                        publisher=publisher,
                        link=item.get("link", ""),
                        published=published,
                        sentiment_label=label,  # type: ignore
                        sentiment_score=score,
                        related_tickers=item.get("relatedTickers", []),
                    )
                    analyzed_articles.append(article)

                except Exception as e:
                    logger.warning(f"Failed to analyze article: {e}")
                    continue

        # Sort by published date (newest first)
        analyzed_articles.sort(key=lambda a: a.published, reverse=True)

        # Trim to max
        analyzed_articles = analyzed_articles[:max_articles]
        total = len(analyzed_articles)

        # Recalculate counts after trimming
        pos_count = sum(1 for a in analyzed_articles if a.sentiment_label == "positive")
        neg_count = sum(1 for a in analyzed_articles if a.sentiment_label == "negative")
        neutral_count = total - pos_count - neg_count

        # Calculate aggregate score (-1 to +1)
        if total > 0:
            aggregate = (pos_count - neg_count) / total
        else:
            aggregate = 0.0

        logger.info(
            f"Sentiment analysis complete for {ticker}: "
            f"{pos_count} positive, {neg_count} negative, {neutral_count} neutral "
            f"(yahoo={yahoo_count}, local={len(local_items)})"
        )

        return SentimentResult(
            ticker=ticker,
            aggregate_score=aggregate,
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neutral_count,
            total_articles=total,
            articles=analyzed_articles,
            analysis_period=period,
            model_used=self._model_label,
        )

    @property
    def _model_label(self) -> str:
        """Return the model name for reporting."""
        if self.use_llm:
            return f"llm ({self.llm_model})"
        elif self.use_vader:
            return "vader"
        elif self.use_hybrid:
            return "finbert-hybrid"
        else:
            return "finbert"

    def get_quick_sentiment(self, ticker: str) -> dict[str, Any]:
        """
        Get quick sentiment summary (lightweight).

        Returns:
            Dict with sentiment score and label
        """
        result = self.analyze(ticker, max_articles=10)
        return {
            "ticker": result.ticker,
            "score": round(result.aggregate_score, 2),
            "sentiment": result.sentiment_label,
            "emoji": result.sentiment_emoji,
            "articles_analyzed": result.total_articles,
        }


def format_sentiment_report(result: SentimentResult) -> str:
    """
    Format sentiment result as a readable string.

    Args:
        result: SentimentResult to format

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("")
    lines.append("=" * 64)
    lines.append(f" NEWS SENTIMENT ANALYSIS: {result.ticker:^38} ")
    lines.append("=" * 64)
    lines.append("")

    # Overall sentiment
    emoji = result.sentiment_emoji
    lines.append(f"Overall Sentiment: {emoji} {result.sentiment_label.upper()}")
    lines.append(f"Aggregate Score: {result.aggregate_score:+.2f} (-1.0 to +1.0)")
    lines.append("")

    # Breakdown
    lines.append("-" * 64)
    lines.append("ARTICLE BREAKDOWN")
    lines.append("-" * 64)
    lines.append(f"   📈 Positive:  {result.positive_count:>3} articles")
    lines.append(f"   📉 Negative:  {result.negative_count:>3} articles")
    lines.append(f"   ⚖️ Neutral:   {result.neutral_count:>3} articles")
    lines.append(f"   Total:        {result.total_articles:>3} articles analyzed")
    lines.append("")

    # Recent articles
    if result.articles:
        lines.append("-" * 64)
        lines.append("RECENT NEWS HEADLINES")
        lines.append("-" * 64)

        for i, article in enumerate(result.articles[:5], 1):
            icon = (
                "📈"
                if article.sentiment_label == "positive"
                else "📉"
                if article.sentiment_label == "negative"
                else "⚖️"
            )
            date_str = article.published.strftime("%m/%d")
            lines.append(f"{i}. {icon} [{date_str}] {article.title[:50]}...")
            lines.append(f"   Source: {article.publisher}")
            lines.append("")

    lines.append("-" * 64)
    lines.append(f"Model: {result.model_used.upper()} | Powered by Yahoo Finance")
    lines.append("")

    return "\n".join(lines)


# Convenience function for CLI usage
def analyze_sentiment(ticker: str, max_articles: int = 20) -> str:
    """
    Quick sentiment analysis for CLI usage.

    Args:
        ticker: Stock ticker
        max_articles: Number of articles to analyze

    Returns:
        Formatted report string
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(ticker, max_articles=max_articles)
    return format_sentiment_report(result)
