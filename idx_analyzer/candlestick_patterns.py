"""
Candlestick Pattern Detection Module for IDX Analyzer.

Detects common candlestick patterns for technical analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd


class PatternType(Enum):
    """Types of candlestick patterns"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class CandlestickPattern:
    """Represents a detected candlestick pattern"""

    name: str
    type: PatternType
    index: int  # Index in the dataframe
    date: pd.Timestamp
    confidence: float  # 0.0 to 1.0
    description: str


def detect_doji(open_price: float, high: float, low: float, close: float) -> bool:
    """
    Detect Doji pattern - open and close are very close.
    Indicates indecision in the market.
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False
    # Doji if body is less than 10% of total range
    return body / total_range < 0.1


def detect_hammer(
    open_price: float, high: float, low: float, close: float
) -> Tuple[bool, bool]:
    """
    Detect Hammer and Inverted Hammer patterns.
    Hammer: Small body at top, long lower shadow (bullish reversal)
    Inverted Hammer: Small body at bottom, long upper shadow (bullish reversal)
    Returns: (is_hammer, is_inverted_hammer)
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False, False

    body_pct = body / total_range
    if body_pct > 0.3:  # Body should be small
        return False, False

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Hammer: long lower shadow, small upper shadow
    is_hammer = lower_shadow > 2 * body and upper_shadow < body

    # Inverted Hammer: long upper shadow, small lower shadow
    is_inverted = upper_shadow > 2 * body and lower_shadow < body

    return is_hammer, is_inverted


def detect_shooting_star(
    open_price: float, high: float, low: float, close: float
) -> bool:
    """
    Detect Shooting Star pattern (bearish reversal).
    Small body at bottom, long upper shadow.
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False

    body_pct = body / total_range
    if body_pct > 0.3:
        return False

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Shooting star: long upper shadow, small/no lower shadow
    return upper_shadow > 2 * body and lower_shadow < body * 0.5


def detect_engulfing(
    prev_open: float,
    prev_high: float,
    prev_low: float,
    prev_close: float,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> Tuple[bool, bool]:
    """
    Detect Bullish and Bearish Engulfing patterns.
    Bullish: Current candle completely engulfs previous bearish candle
    Bearish: Current candle completely engulfs previous bullish candle
    Returns: (is_bullish_engulfing, is_bearish_engulfing)
    """
    prev_body = abs(prev_close - prev_open)
    curr_body = abs(close - open_price)

    if prev_body == 0 or curr_body == 0:
        return False, False

    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open
    curr_bullish = close > open_price
    curr_bearish = close < open_price

    # Bullish engulfing: current bullish candle engulfs previous bearish
    is_bullish = (
        curr_bullish
        and prev_bearish
        and open_price <= prev_close
        and close >= prev_open
    )

    # Bearish engulfing: current bearish candle engulfs previous bullish
    is_bearish = (
        curr_bearish
        and prev_bullish
        and open_price >= prev_close
        and close <= prev_open
    )

    return is_bullish, is_bearish


def detect_morning_star(df: pd.DataFrame, idx: int, lookback: int = 1) -> bool:
    """
    Detect Morning Star pattern (bullish reversal).
    Day 1: Long bearish candle
    Day 2: Small body (gap down)
    Day 3: Long bullish candle that closes well into Day 1
    """
    if idx < 2:
        return False

    i = idx - lookback
    if i < 2:
        return False

    # Day 1 - Bearish
    day1_open = df["Open"].iloc[i - 2]
    day1_close = df["Close"].iloc[i - 2]
    day1_bearish = day1_close < day1_open

    # Day 2 - Small body
    day2_open = df["Open"].iloc[i - 1]
    day2_close = df["Close"].iloc[i - 1]
    day2_body = abs(day2_close - day2_open)
    day2_range = df["High"].iloc[i - 1] - df["Low"].iloc[i - 1]
    day2_small = day2_body < day2_range * 0.3 if day2_range > 0 else False

    # Day 3 - Bullish, closes into Day 1
    day3_open = df["Open"].iloc[i]
    day3_close = df["Close"].iloc[i]
    day3_bullish = day3_close > day3_open
    day3_strong = day3_close > (day1_open + day1_close) / 2

    return day1_bearish and day2_small and day3_bullish and day3_strong


def detect_evening_star(df: pd.DataFrame, idx: int, lookback: int = 1) -> bool:
    """
    Detect Evening Star pattern (bearish reversal).
    Day 1: Long bullish candle
    Day 2: Small body (gap up)
    Day 3: Long bearish candle that closes well into Day 1
    """
    if idx < 2:
        return False

    i = idx - lookback
    if i < 2:
        return False

    # Day 1 - Bullish
    day1_open = df["Open"].iloc[i - 2]
    day1_close = df["Close"].iloc[i - 2]
    day1_bullish = day1_close > day1_open

    # Day 2 - Small body
    day2_open = df["Open"].iloc[i - 1]
    day2_close = df["Close"].iloc[i - 1]
    day2_body = abs(day2_close - day2_open)
    day2_range = df["High"].iloc[i - 1] - df["Low"].iloc[i - 1]
    day2_small = day2_body < day2_range * 0.3 if day2_range > 0 else False

    # Day 3 - Bearish, closes into Day 1
    day3_open = df["Open"].iloc[i]
    day3_close = df["Close"].iloc[i]
    day3_bearish = day3_close < day3_open
    day3_strong = day3_close < (day1_open + day1_close) / 2

    return day1_bullish and day2_small and day3_bearish and day3_strong


def detect_three_white_soldiers(df: pd.DataFrame, idx: int) -> bool:
    """
    Detect Three White Soldiers (bullish continuation).
    Three consecutive bullish candles with higher closes.
    """
    if idx < 2:
        return False

    for i in range(idx - 2, idx + 1):
        if df["Close"].iloc[i] <= df["Open"].iloc[i]:
            return False

    # Each close higher than previous
    for i in range(idx - 1, idx + 1):
        if df["Close"].iloc[i] <= df["Close"].iloc[i - 1]:
            return False

    return True


def detect_three_black_crows(df: pd.DataFrame, idx: int) -> bool:
    """
    Detect Three Black Crows (bearish continuation).
    Three consecutive bearish candles with lower closes.
    """
    if idx < 2:
        return False

    for i in range(idx - 2, idx + 1):
        if df["Close"].iloc[i] >= df["Open"].iloc[i]:
            return False

    # Each close lower than previous
    for i in range(idx - 1, idx + 1):
        if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]:
            return False

    return True


def detect_spinning_top(
    open_price: float, high: float, low: float, close: float
) -> bool:
    """
    Detect Spinning Top (indecision).
    Small body with long upper and lower shadows.
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False

    body_pct = body / total_range
    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Small body, both shadows significant
    return body_pct < 0.2 and upper_shadow > body * 1.5 and lower_shadow > body * 1.5


def detect_marubozu(
    open_price: float, high: float, low: float, close: float
) -> Tuple[bool, bool]:
    """
    Detect Marubozu (strong trend).
    Full body with no shadows.
    Returns: (is_bullish, is_bearish)
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False, False

    body_pct = body / total_range
    if body_pct < 0.95:  # Must be almost full range
        return False, False

    is_bullish = close > open_price
    is_bearish = close < open_price

    return is_bullish, is_bearish


def detect_hanging_man(
    open_price: float, high: float, low: float, close: float
) -> bool:
    """
    Detect Hanging Man (bearish reversal at top).
    Similar to hammer but appears after uptrend.
    """
    body = abs(close - open_price)
    total_range = high - low
    if total_range == 0:
        return False

    body_pct = body / total_range
    if body_pct > 0.3:
        return False

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # Long lower shadow, small upper shadow, body at top
    return lower_shadow > 2 * body and upper_shadow < body


def detect_all_patterns(df: pd.DataFrame) -> List[CandlestickPattern]:
    """
    Detect all candlestick patterns in the dataframe.
    Returns list of detected patterns for the most recent candles.
    """
    patterns = []

    if len(df) < 3:
        return patterns

    # Focus on recent candles (last 20)
    start_idx = max(0, len(df) - 20)

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        open_p = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        date = df.index[idx]

        # Single candle patterns
        if detect_doji(open_p, high, low, close):
            patterns.append(
                CandlestickPattern(
                    name="Doji",
                    type=PatternType.NEUTRAL,
                    index=idx,
                    date=date,
                    confidence=0.7,
                    description="Indecision - market is balanced",
                )
            )

        hammer, inverted = detect_hammer(open_p, high, low, close)
        if hammer:
            patterns.append(
                CandlestickPattern(
                    name="Hammer",
                    type=PatternType.BULLISH,
                    index=idx,
                    date=date,
                    confidence=0.75,
                    description="Potential bullish reversal",
                )
            )
        if inverted:
            patterns.append(
                CandlestickPattern(
                    name="Inverted Hammer",
                    type=PatternType.BULLISH,
                    index=idx,
                    date=date,
                    confidence=0.7,
                    description="Potential bullish reversal",
                )
            )

        if detect_shooting_star(open_p, high, low, close):
            patterns.append(
                CandlestickPattern(
                    name="Shooting Star",
                    type=PatternType.BEARISH,
                    index=idx,
                    date=date,
                    confidence=0.75,
                    description="Potential bearish reversal",
                )
            )

        if detect_spinning_top(open_p, high, low, close):
            patterns.append(
                CandlestickPattern(
                    name="Spinning Top",
                    type=PatternType.NEUTRAL,
                    index=idx,
                    date=date,
                    confidence=0.6,
                    description="Indecision - possible trend change",
                )
            )

        maru_bull, maru_bear = detect_marubozu(open_p, high, low, close)
        if maru_bull:
            patterns.append(
                CandlestickPattern(
                    name="Bullish Marubozu",
                    type=PatternType.BULLISH,
                    index=idx,
                    date=date,
                    confidence=0.85,
                    description="Strong bullish momentum",
                )
            )
        if maru_bear:
            patterns.append(
                CandlestickPattern(
                    name="Bearish Marubozu",
                    type=PatternType.BEARISH,
                    index=idx,
                    date=date,
                    confidence=0.85,
                    description="Strong bearish momentum",
                )
            )

        if detect_hanging_man(open_p, high, low, close):
            patterns.append(
                CandlestickPattern(
                    name="Hanging Man",
                    type=PatternType.BEARISH,
                    index=idx,
                    date=date,
                    confidence=0.7,
                    description="Potential bearish reversal at top",
                )
            )

        # Multi-candle patterns (need previous candle)
        if idx > 0:
            prev_row = df.iloc[idx - 1]
            prev_open = float(prev_row["Open"])
            prev_high = float(prev_row["High"])
            prev_low = float(prev_row["Low"])
            prev_close = float(prev_row["Close"])

            bull_engulf, bear_engulf = detect_engulfing(
                prev_open, prev_high, prev_low, prev_close, open_p, high, low, close
            )
            if bull_engulf:
                patterns.append(
                    CandlestickPattern(
                        name="Bullish Engulfing",
                        type=PatternType.BULLISH,
                        index=idx,
                        date=date,
                        confidence=0.8,
                        description="Strong bullish reversal signal",
                    )
                )
            if bear_engulf:
                patterns.append(
                    CandlestickPattern(
                        name="Bearish Engulfing",
                        type=PatternType.BEARISH,
                        index=idx,
                        date=date,
                        confidence=0.8,
                        description="Strong bearish reversal signal",
                    )
                )

        # Three-candle patterns
        if detect_morning_star(df, idx):
            patterns.append(
                CandlestickPattern(
                    name="Morning Star",
                    type=PatternType.BULLISH,
                    index=idx,
                    date=date,
                    confidence=0.85,
                    description="Strong bullish reversal pattern",
                )
            )

        if detect_evening_star(df, idx):
            patterns.append(
                CandlestickPattern(
                    name="Evening Star",
                    type=PatternType.BEARISH,
                    index=idx,
                    date=date,
                    confidence=0.85,
                    description="Strong bearish reversal pattern",
                )
            )

        if detect_three_white_soldiers(df, idx):
            patterns.append(
                CandlestickPattern(
                    name="Three White Soldiers",
                    type=PatternType.BULLISH,
                    index=idx,
                    date=date,
                    confidence=0.8,
                    description="Bullish continuation pattern",
                )
            )

        if detect_three_black_crows(df, idx):
            patterns.append(
                CandlestickPattern(
                    name="Three Black Crows",
                    type=PatternType.BEARISH,
                    index=idx,
                    date=date,
                    confidence=0.8,
                    description="Bearish continuation pattern",
                )
            )

    return patterns


def get_recent_patterns(
    df: pd.DataFrame, max_patterns: int = 5
) -> List[CandlestickPattern]:
    """
    Get the most recent significant patterns.
    Returns patterns from the last 5 candles, sorted by confidence.
    """
    all_patterns = detect_all_patterns(df)

    if not all_patterns:
        return []

    # Filter to last 5 candles
    last_idx = len(df) - 1
    recent = [p for p in all_patterns if p.index >= last_idx - 5]

    # Sort by index (newest first) then by confidence
    recent.sort(key=lambda p: (p.index, p.confidence), reverse=True)

    # Return unique pattern names (keep highest confidence for duplicates)
    seen_names = set()
    unique_patterns = []
    for p in recent:
        if p.name not in seen_names:
            seen_names.add(p.name)
            unique_patterns.append(p)

    return unique_patterns[:max_patterns]


def format_patterns_for_display(patterns: List[CandlestickPattern]) -> str:
    """Format patterns for display in terminal"""
    if not patterns:
        return "No significant patterns detected"

    icon_map = {
        PatternType.BULLISH: "[B]",
        PatternType.BEARISH: "[S]",
        PatternType.NEUTRAL: "[N]",
    }

    lines = []
    for p in patterns:
        icon = icon_map.get(p.type, "◆")
        lines.append(f"{icon} {p.name} ({p.confidence * 100:.0f}% confidence)")
        lines.append(f"   {p.description}")

    return "\n".join(lines)
