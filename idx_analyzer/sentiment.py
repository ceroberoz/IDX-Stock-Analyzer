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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

import logging

logger = logging.getLogger(__name__)


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
    model_used: str = "finbert"

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

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("BBCA.JK")
        >>> print(f"Sentiment: {result.sentiment_label} ({result.aggregate_score:+.2f})")
    """

    def __init__(self, model_name: str = "ProsusAI/finBERT", use_vader: bool = False):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finBERT)
            use_vader: If True, use VADER instead of FinBERT (no download, lighter)
        """
        self.model_name = model_name
        self.use_vader = use_vader
        self._pipeline = None
        self._analyzer = None

    def _get_pipeline(self):
        """Lazy load the NLP pipeline"""
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

    def _analyze_text(self, text: str) -> tuple[str, float]:
        """
        Analyze single text and return sentiment label and score.

        Returns:
            Tuple of (label, confidence_score)
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
            return label, confidence
        else:
            # FinBERT analysis
            pipeline = self._get_pipeline()
            result = pipeline(text)[0]
            return result["label"].lower(), result["score"]

    def analyze(
        self,
        ticker: str,
        max_articles: int = 20,
        period: str = "7d",
    ) -> SentimentResult:
        """
        Analyze sentiment for a stock ticker.

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
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        # Initialize pipeline (will download model on first run)
        self._get_pipeline()

        # Fetch news from Yahoo Finance
        logger.info(f"Fetching news for {ticker}")
        stock = yf.Ticker(ticker)

        try:
            news_items = stock.get_news(count=max_articles, tab="news")
        except Exception as e:
            logger.warning(f"get_news() failed: {e}, trying .news property")
            news_items = stock.news[:max_articles] if stock.news else []

        if not news_items:
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
                model_used="vader" if self.use_vader else "finbert",
            )

        # Analyze each article
        analyzed_articles: list[NewsArticle] = []
        pos_count = neg_count = neutral_count = 0

        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue

            try:
                label, score = self._analyze_text(title)

                # Parse timestamp
                published_ts = item.get("published") or item.get("date", 0)
                try:
                    published = datetime.fromtimestamp(published_ts)
                except (ValueError, TypeError, OSError):
                    published = datetime.now()

                article = NewsArticle(
                    title=title,
                    publisher=item.get("publisher", "Unknown"),
                    link=item.get("link", ""),
                    published=published,
                    sentiment_label=label,  # type: ignore
                    sentiment_score=score,
                    related_tickers=item.get("relatedTickers", []),
                )
                analyzed_articles.append(article)

                # Count sentiments
                if label == "positive":
                    pos_count += 1
                elif label == "negative":
                    neg_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                logger.warning(f"Failed to analyze article: {e}")
                continue

        # Calculate aggregate score (-1 to +1)
        total = len(analyzed_articles)
        if total > 0:
            # Weight: positive=+1, negative=-1, neutral=0
            aggregate = (pos_count - neg_count) / total
        else:
            aggregate = 0.0

        logger.info(
            f"Sentiment analysis complete for {ticker}: "
            f"{pos_count} positive, {neg_count} negative, {neutral_count} neutral"
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
            model_used="vader" if self.use_vader else "finbert",
        )

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
