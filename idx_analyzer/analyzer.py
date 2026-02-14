"""
Core analysis module for IDX stocks with enhanced error handling and configuration support.
"""

import re
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .config import Config, get_config
from .exceptions import (
    AnalysisError,
    ChartError,
    DataFetchError,
    InsufficientDataError,
    InvalidTickerError,
    NetworkError,
)


class SupportResistance:
    """Support and resistance levels"""

    def __init__(self, level: float, type: str, strength: str, description: str):
        self.level = float(level)
        self.type = type
        self.strength = strength
        self.description = description


class AnalysisResult:
    """Complete analysis result"""

    def __init__(
        self,
        ticker: str,
        current_price: float,
        change_percent: float,
        volume: int,
        week_52_high: float,
        week_52_low: float,
        support_levels: List[SupportResistance],
        resistance_levels: List[SupportResistance],
        trend: str,
        recommendation: str,
        summary: str,
        rsi: float,
        sma_20: float,
        sma_50: float,
        sma_200: Optional[float],
        bb_middle: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        bb_position: Optional[str] = None,
        vp_poc: Optional[float] = None,
        vp_value_area_high: Optional[float] = None,
        vp_value_area_low: Optional[float] = None,
        vp_total_volume: Optional[float] = None,
        market_cap: Optional[float] = None,
        pe_ratio: Optional[float] = None,
        dividend_yield: Optional[float] = None,
    ):
        self.ticker = ticker
        self.current_price = current_price
        self.change_percent = change_percent
        self.volume = volume
        self.week_52_high = week_52_high
        self.week_52_low = week_52_low
        self.support_levels = support_levels
        self.resistance_levels = resistance_levels
        self.trend = trend
        self.recommendation = recommendation
        self.summary = summary
        self.rsi = rsi
        self.sma_20 = sma_20
        self.sma_50 = sma_50
        self.sma_200 = sma_200
        self.bb_middle = bb_middle
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_position = bb_position
        self.vp_poc = vp_poc
        self.vp_value_area_high = vp_value_area_high
        self.vp_value_area_low = vp_value_area_low
        self.vp_total_volume = vp_total_volume
        self.market_cap = market_cap
        self.pe_ratio = pe_ratio
        self.dividend_yield = dividend_yield


def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to add retry logic with exponential backoff"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2**attempt))
            raise last_exception

        return wrapper

    return decorator


class IDXAnalyzer:
    """Analyzer for Indonesian stocks (IDX)"""

    def __init__(self, ticker: str, config: Optional[Config] = None):
        self.ticker = self._format_ticker(ticker)
        self.config = config or get_config()
        self.stock = yf.Ticker(self.ticker)
        self.hist: Optional[pd.DataFrame] = None
        self.info: Optional[dict] = None

    def _format_ticker(self, ticker: str) -> str:
        """Format ticker to Yahoo Finance format"""
        if ticker.upper().startswith("IDX:"):
            ticker = ticker[4:]
        if not ticker.upper().endswith(".JK"):
            ticker = f"{ticker}.JK"
        return ticker.upper()

    @with_retry()
    def _fetch_with_retry(self, period: str) -> Tuple[pd.DataFrame, dict]:
        """Fetch data with retry logic"""
        hist = self.stock.history(period=period)
        info = {}
        try:
            info = self.stock.info or {}
        except Exception:
            pass
        return hist, info

    def fetch_data(self, period: Optional[str] = None) -> bool:
        """Fetch historical data and stock info with error handling"""
        if period is None:
            period = self.config.analysis.default_period

        try:
            self.hist, self.info = self._fetch_with_retry(period)

            if self.hist is None or len(self.hist) == 0:
                raise InvalidTickerError(
                    self.ticker.replace(".JK", ""),
                    "No data returned. The ticker may not exist or may be delisted.",
                )

            if len(self.hist) < 20:
                raise InsufficientDataError(
                    ticker=self.ticker.replace(".JK", ""),
                    data_points=len(self.hist),
                    required=20,
                    details="Not enough data points for reliable analysis",
                )

            return True

        except InvalidTickerError:
            raise
        except InsufficientDataError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(
                x in error_msg
                for x in ["connection", "timeout", "network", "urllib", "http"]
            ):
                raise NetworkError(
                    message="Failed to connect to Yahoo Finance",
                    ticker=self.ticker.replace(".JK", ""),
                    retry_count=self.config.network.max_retries,
                    details=str(e),
                )
            raise DataFetchError(
                message=f"Failed to fetch data: {e}",
                ticker=self.ticker.replace(".JK", ""),
                details=str(e),
            )

    def _calculate_rsi(self, window: Optional[int] = None) -> pd.Series:
        """Calculate Relative Strength Index"""
        if window is None:
            window = self.config.analysis.rsi_window

        if self.hist is None or len(self.hist) < window:
            raise AnalysisError(
                "Cannot calculate RSI",
                f"Need at least {window} data points, have {len(self.hist) if self.hist else 0}",
            )

        delta = self.hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        if self.hist is None or len(self.hist) < window:
            raise AnalysisError(
                f"Cannot calculate SMA {window}",
                f"Need at least {window} data points, have {len(self.hist) if self.hist else 0}",
            )
        return self.hist["Close"].rolling(window=window).mean()

    def _calculate_bollinger_bands(
        self, window: Optional[int] = None, num_std: Optional[float] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        if window is None:
            window = self.config.analysis.bb_window
        if num_std is None:
            num_std = self.config.analysis.bb_std

        if self.hist is None or len(self.hist) < window:
            raise AnalysisError(
                "Cannot calculate Bollinger Bands",
                f"Need at least {window} data points, have {len(self.hist) if self.hist else 0}",
            )

        middle = self._calculate_sma(window)
        std = self.hist["Close"].rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return middle, upper, lower

    def _calculate_volume_profile(
        self, num_bins: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate Volume Profile"""
        if num_bins is None:
            num_bins = self.config.analysis.vp_bins

        if self.hist is None or len(self.hist) < 20:
            return None, None, None, None

        price_low = float(self.hist["Low"].min())
        price_high = float(self.hist["High"].max())

        if price_high == price_low:
            return price_low, price_low, price_low, float(self.hist["Volume"].sum())

        bins = np.linspace(price_low, price_high, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_by_bin = np.zeros(num_bins)

        for i in range(len(self.hist)):
            row = self.hist.iloc[i]
            candle_low = float(row["Low"])
            candle_high = float(row["High"])
            candle_volume = float(row["Volume"])

            low_bin_idx = np.searchsorted(bins, candle_low, side="left") - 1
            high_bin_idx = np.searchsorted(bins, candle_high, side="right") - 1

            low_bin_idx = max(0, min(low_bin_idx, num_bins - 1))
            high_bin_idx = max(0, min(high_bin_idx, num_bins - 1))

            if low_bin_idx == high_bin_idx:
                volume_by_bin[low_bin_idx] += candle_volume
            else:
                bins_in_range = high_bin_idx - low_bin_idx + 1
                volume_per_bin = candle_volume / bins_in_range
                for j in range(low_bin_idx, high_bin_idx + 1):
                    volume_by_bin[j] += volume_per_bin

        total_volume = volume_by_bin.sum()
        if total_volume == 0:
            return None, None, None, None

        poc_idx = int(np.argmax(volume_by_bin))
        poc = float(bin_centers[poc_idx])

        sorted_indices = np.argsort(volume_by_bin)[::-1]
        cumulative_volume = 0.0
        value_area_indices = []
        target_volume = total_volume * 0.70

        for idx in sorted_indices:
            cumulative_volume += volume_by_bin[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break

        if value_area_indices:
            value_area_low_idx = min(value_area_indices)
            value_area_high_idx = max(value_area_indices)
            value_area_low = float(bins[value_area_low_idx])
            value_area_high = float(
                bins[value_area_high_idx + 1]
                if value_area_high_idx + 1 < len(bins)
                else bins[value_area_high_idx]
            )
        else:
            value_area_low = price_low
            value_area_high = price_high

        return poc, value_area_high, value_area_low, float(total_volume)

    def _find_support_levels(self) -> List[SupportResistance]:
        """Find key support levels"""
        supports = []
        if self.hist is None or len(self.hist) == 0:
            return supports

        low_52w = float(self.hist["Low"].min())
        supports.append(
            SupportResistance(
                level=round(low_52w, 0),
                type="support",
                strength="strong",
                description="52-week low",
            )
        )

        recent = self.hist.tail(30)
        recent_lows = recent.nsmallest(3, "Low")["Low"].values
        for low in recent_lows:
            if float(low) > low_52w * 1.02:
                supports.append(
                    SupportResistance(
                        level=round(float(low), 0),
                        type="support",
                        strength="moderate",
                        description="Recent swing low",
                    )
                )

        current = float(self.hist["Close"].iloc[-1])
        round_levels = [int(current / 100) * 100, int(current / 500) * 500]
        for level in set(round_levels):
            if level < current and level > low_52w:
                supports.append(
                    SupportResistance(
                        level=float(level),
                        type="support",
                        strength="weak",
                        description="Psychological level",
                    )
                )

        seen = set()
        unique_supports = []
        for s in sorted(supports, key=lambda x: x.level, reverse=True):
            if s.level not in seen:
                seen.add(s.level)
                unique_supports.append(s)
        return unique_supports[:4]

    def _find_resistance_levels(self) -> List[SupportResistance]:
        """Find key resistance levels"""
        resistances = []
        if self.hist is None or len(self.hist) == 0:
            return resistances

        current = float(self.hist["Close"].iloc[-1])
        high_52w = float(self.hist["High"].max())
        if high_52w > current * 1.01:
            resistances.append(
                SupportResistance(
                    level=round(high_52w, 0),
                    type="resistance",
                    strength="strong",
                    description="52-week high",
                )
            )

        recent = self.hist.tail(30)
        recent_highs = recent.nlargest(3, "High")["High"].values
        for high in recent_highs:
            if float(high) > current * 1.01:
                resistances.append(
                    SupportResistance(
                        level=round(float(high), 0),
                        type="resistance",
                        strength="moderate",
                        description="Recent swing high",
                    )
                )

        round_levels = [
            int((current + 500) / 500) * 500,
            int((current + 1000) / 1000) * 1000,
        ]
        for level in set(round_levels):
            if level > current:
                resistances.append(
                    SupportResistance(
                        level=float(level),
                        type="resistance",
                        strength="weak",
                        description="Psychological level",
                    )
                )

        seen = set()
        unique_resistances = []
        for r in sorted(resistances, key=lambda x: x.level):
            if r.level not in seen:
                seen.add(r.level)
                unique_resistances.append(r)
        return unique_resistances[:4]

    def _determine_trend(
        self, rsi: float, sma_20: float, sma_50: float, sma_200: Optional[float]
    ) -> str:
        """Determine trend based on technical indicators"""
        if self.hist is None or len(self.hist) < 20:
            return "Unknown"

        current = float(self.hist["Close"].iloc[-1])

        trend_signals = []
        if current > sma_20:
            trend_signals.append("Above SMA 20")
        if current > sma_50:
            trend_signals.append("Above SMA 50")
        if sma_200 and current > sma_200:
            trend_signals.append("Above SMA 200")

        if len(trend_signals) == 3:
            trend = "Bullish (Strong Uptrend)"
        elif len(trend_signals) == 2:
            trend = "Bullish"
        elif len(trend_signals) == 1:
            trend = "Neutral (Mixed)"
        else:
            trend = "Bearish"

        if sma_200 and sma_50 > sma_200:
            if "Bullish" in trend:
                trend = "Bullish (Golden Cross Active)"
        elif sma_200 and sma_50 < sma_200:
            if "Bearish" in trend:
                trend = "Bearish (Death Cross Active)"

        if rsi > 70:
            trend += " | Overbought"
        elif rsi < 30:
            trend += " | Oversold"

        return trend

    def _generate_recommendation(
        self,
        trend: str,
        rsi: float,
        supports: List[SupportResistance],
        resistances: List[SupportResistance],
    ) -> str:
        """Generate trading recommendation"""
        if not resistances or not supports:
            return "Consolidating. Wait for clear signal."

        current = float(self.hist["Close"].iloc[-1])
        nearest_r = min([r.level for r in resistances])
        nearest_s = max([s.level for s in supports])

        if "Bullish" in trend:
            if rsi > 75:
                return f"Overextended. Potential pullback to {nearest_s:,.0f}."
            return f"Bullish. Target: {nearest_r:,.0f} (+{(nearest_r - current) / current * 100:.1f}%)."
        elif "Bearish" in trend:
            if rsi < 25:
                return f"Oversold. Potential bounce from {nearest_s:,.0f}."
            return f"Bearish. Support at {nearest_s:,.0f} ({(current - nearest_s) / current * 100:.1f}% below)."
        return f"Range-bound ({nearest_s:,.0f} - {nearest_r:,.0f})."

    def analyze(self) -> AnalysisResult:
        """Perform full technical analysis"""
        if self.hist is None:
            raise AnalysisError(
                "No data available",
                "Call fetch_data() before analyze()",
            )

        try:
            current = float(self.hist["Close"].iloc[-1])
            prev_close = (
                float(self.hist["Close"].iloc[-2]) if len(self.hist) > 1 else current
            )
            change_pct = (current - prev_close) / prev_close * 100

            rsi_series = self._calculate_rsi()
            current_rsi = float(rsi_series.iloc[-1])

            sma_20_series = self._calculate_sma(20)
            sma_50_series = self._calculate_sma(50)
            sma_200_series = None
            if len(self.hist) >= 200:
                sma_200_series = self._calculate_sma(200)

            sma_20 = float(sma_20_series.iloc[-1])
            sma_50 = float(sma_50_series.iloc[-1])
            sma_200 = (
                float(sma_200_series.iloc[-1]) if sma_200_series is not None else None
            )

            bb_middle_series, bb_upper_series, bb_lower_series = (
                self._calculate_bollinger_bands()
            )
            bb_middle = float(bb_middle_series.iloc[-1])
            bb_upper = float(bb_upper_series.iloc[-1])
            bb_lower = float(bb_lower_series.iloc[-1])

            bb_band_width = bb_upper - bb_lower
            bb_position = "middle"
            if bb_band_width > 0:
                if current > bb_upper:
                    bb_position = "above_upper"
                elif current > bb_upper - bb_band_width * 0.2:
                    bb_position = "near_upper"
                elif current < bb_lower:
                    bb_position = "below_lower"
                elif current < bb_lower + bb_band_width * 0.2:
                    bb_position = "near_lower"

            vp_poc, vp_va_high, vp_va_low, vp_total = self._calculate_volume_profile()

            supports = self._find_support_levels()
            resistances = self._find_resistance_levels()
            trend = self._determine_trend(current_rsi, sma_20, sma_50, sma_200)
            recommendation = self._generate_recommendation(
                trend, current_rsi, supports, resistances
            )

            info = self.info or {}

            return AnalysisResult(
                ticker=self.ticker.replace(".JK", ""),
                current_price=current,
                change_percent=change_pct,
                volume=int(self.hist["Volume"].iloc[-1]),
                week_52_high=float(self.hist["High"].max()),
                week_52_low=float(self.hist["Low"].min()),
                support_levels=supports,
                resistance_levels=resistances,
                trend=trend,
                recommendation=recommendation,
                summary=self._generate_summary(
                    current, trend, current_rsi, supports, resistances
                ),
                rsi=current_rsi,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                bb_middle=bb_middle,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_position=bb_position,
                vp_poc=vp_poc,
                vp_value_area_high=vp_va_high,
                vp_value_area_low=vp_va_low,
                vp_total_volume=vp_total,
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                dividend_yield=info.get("dividendYield"),
            )

        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(
                "Analysis failed",
                str(e),
            )

    def _generate_summary(
        self,
        current: float,
        trend: str,
        rsi: float,
        supports: List[SupportResistance],
        resistances: List[SupportResistance],
    ) -> str:
        """Generate text summary"""
        info = self.info or {}
        market_cap = info.get("marketCap", 0)
        pe = info.get("trailingPE", 0)

        lines = [
            f"Price: {current:,.0f} | RSI: {rsi:.1f} | Trend: {trend}",
        ]

        if market_cap:
            lines.append(f"MC: {market_cap / 1e12:.1f}T | P/E: {pe:.1f}")

        return " | ".join(lines)

    def _generate_narrative(
        self,
        result: AnalysisResult,
        trend_emoji: str,
        nearest_support: float,
        nearest_resistance: float,
        risk_reward: float,
    ) -> str:
        """Generate playful, emoji-rich narrative text"""
        current = result.current_price
        rsi = result.rsi
        trend = result.trend

        # 1. Headline
        headline_icon = "💎"
        if "Bull" in trend:
            headline_icon = "🚀"
        elif "Bear" in trend:
            headline_icon = "🐻"

        sections = [f"{headline_icon} {result.ticker} @ {current:,.0f}"]

        # 2. Price Context
        from_low = (current - result.week_52_low) / result.week_52_low * 100
        from_high = (current - result.week_52_high) / result.week_52_high * 100

        if "Bull" in trend:
            price_ctx = f"🎢 Climbing! +{from_low:.1f}% from the bottom"
            if from_high > -20:
                price_ctx += f", almost at the peak! ({-from_high:.1f}% to go)"
        elif "Bear" in trend:
            price_ctx = f"📉 Discount Mode: {-from_high:.1f}% off the highs"
        else:
            price_ctx = f"⚖️ Chilling in the middle of the range"
        sections.append(price_ctx)

        # 3. Trend Analysis
        if "Golden Cross" in trend:
            trend_text = "✨ GOLDEN CROSS! SMA 50 > 200 (Bull Mode Activated)"
        elif "Death Cross" in trend:
            trend_text = "☠️ DEATH CROSS! SMA 50 < 200 (Bears are feasting)"
        elif "Bullish" in trend:
            trend_text = "🐂 Bulls in Control: Price surfing above MAs"
        elif "Bearish" in trend:
            trend_text = "🧊 Ice Cold: Price trapped below key MAs"
        else:
            trend_text = "🤷 Mixed Signals: Market is confused right now"
        sections.append(trend_text)

        # 4. Momentum (RSI)
        if rsi > 70:
            rsi_text = f"🔥 Too Hot to Handle! (RSI {rsi:.1f}) - Pullback incoming?"
        elif rsi < 30:
            rsi_text = f"🥶 Deep Freeze (RSI {rsi:.1f}) - Bounce imminent?"
        elif 50 < rsi <= 70:
            rsi_text = f"🔋 Charged Up (RSI {rsi:.1f}) - Bullish vibes"
        elif 30 <= rsi < 50:
            rsi_text = f"🥀 Losing Steam (RSI {rsi:.1f}) - Bearish vibes"
        else:
            rsi_text = f"😐 Neutral Zone (RSI {rsi:.1f}) - Waiting for a spark"
        sections.append(rsi_text)

        # 5. Support/Resistance
        support_dist = (current - nearest_support) / current * 100
        resist_dist = (nearest_resistance - current) / current * 100

        sr_text = f"🛡️ Safety Net: {nearest_support:,.0f} (-{support_dist:.1f}%)"
        if risk_reward > 0:
            sr_text += f" | 🧱 Wall: {nearest_resistance:,.0f} (+{resist_dist:.1f}%)"
            sr_text += f"\n💰 Risk/Reward Ratio: 1:{risk_reward:.1f}"
        sections.append(sr_text)

        # 6. Action
        if "Golden Cross" in trend and rsi < 70:
            action = "🚀 BUY SIGNAL: Trend is your friend, ride the wave!"
        elif "Death Cross" in trend and rsi > 30:
            action = "⛔ SELL/AVOID: Don't catch a falling knife"
        elif "Bull" in trend and rsi < 30:
            action = "🛒 BUY THE DIP: Strong trend + Oversold = Opportunity"
        elif "Bear" in trend and rsi > 70:
            action = "💸 SELL THE RALLY: Bear trend + Overbought = Trap"
        elif rsi > 75:
            action = "🤑 TAKE PROFITS: Market is screaming 'Too High!'"
        elif rsi < 25:
            action = "👀 WATCH CLOSELY: Reversal could happen any second"
        elif "Bull" in trend:
            action = "✊ HOLD/BUY: Stay long, add on support"
        elif "Bear" in trend:
            action = "💤 AVOID/HOLD: Cash is king right now"
        else:
            action = "🍿 WAIT & WATCH: Let the market decide first"
        sections.append(f"Action: {action}")

        return "\n".join(sections)

    def generate_rich_report(self, result: AnalysisResult):
        """Generate a rich text narrative for CLI display"""
        from rich import box
        from rich.align import Align
        from rich.columns import Columns
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # 1. Header Section
        change_color = "green" if result.change_percent >= 0 else "red"
        price_text = Text(f"{result.current_price:,.0f}", style=f"bold {change_color}")
        change_text = Text(f"{result.change_percent:+.2f}%", style=change_color)

        header_grid = Table.grid(expand=True)
        header_grid.add_column(justify="left")
        header_grid.add_column(justify="right")
        header_grid.add_row(
            Text(
                f"📊 Market Intel: {result.ticker}", style="bold white", justify="left"
            ),
            Text.assemble("Price: ", price_text, " (", change_text, ")"),
        )

        # 2. Trend & Momentum Section
        trend_color = (
            "green"
            if "Bull" in result.trend
            else "red"
            if "Bear" in result.trend
            else "yellow"
        )

        rsi_color = "red" if result.rsi > 70 else "green" if result.rsi < 30 else "blue"
        rsi_status = (
            "🔥 Overbought"
            if result.rsi > 70
            else "🥶 Oversold"
            if result.rsi < 30
            else "⚖️ Neutral"
        )

        metrics_table = Table(box=None, padding=(0, 2), expand=True)
        metrics_table.add_column("🚀 Metric", style="cyan")
        metrics_table.add_column("💎 Value", style="bold white")
        metrics_table.add_column("🚦 Status", justify="right")

        metrics_table.add_row(
            "Trend", result.trend.split(" (")[0], Text(result.trend, style=trend_color)
        )
        metrics_table.add_row(
            "RSI (14)", f"{result.rsi:.1f}", Text(f"{rsi_status}", style=rsi_color)
        )

        # Moving Averages
        ma_status = []
        if result.current_price > result.sma_20:
            ma_status.append("[green]📈 >SMA20[/green]")
        else:
            ma_status.append("[red]📉 <SMA20[/red]")

        if result.current_price > result.sma_50:
            ma_status.append("[green]📈 >SMA50[/green]")
        else:
            ma_status.append("[red]📉 <SMA50[/red]")

        if result.sma_200:
            if result.current_price > result.sma_200:
                ma_status.append("[green]📈 >SMA200[/green]")
            else:
                ma_status.append("[red]📉 <SMA200[/red]")

        metrics_table.add_row("Mov. Avgs", ", ".join(ma_status), "")

        # 3. Key Levels Section
        levels_table = Table(
            title="🧱 Support & Resistance Zones",
            box=box.SIMPLE,
            expand=True,
            title_style="bold yellow",
        )
        levels_table.add_column("Type", style="cyan")
        levels_table.add_column("Level", justify="right", style="bold white")
        levels_table.add_column("Distance", justify="right")
        levels_table.add_column("Strength", style="dim white")

        # Supports
        for s in result.support_levels[:2]:
            dist = (result.current_price - s.level) / result.current_price * 100
            levels_table.add_row(
                "🛡️ Support",
                f"{s.level:,.0f}",
                f"[red]{dist:.1f}% below[/red]",
                s.strength.capitalize(),
            )

        # Resistances
        for r in result.resistance_levels[:2]:
            dist = (r.level - result.current_price) / result.current_price * 100
            levels_table.add_row(
                "🧱 Resistance",
                f"{r.level:,.0f}",
                f"[green]+{dist:.1f}% above[/green]",
                r.strength.capitalize(),
            )

        # 4. Recommendation Section
        rec_color = "white"
        rec_icon = "🤔"
        if "BUY" in result.recommendation.upper():
            rec_color = "bold green"
            rec_icon = "🚀"
        elif "SELL" in result.recommendation.upper():
            rec_color = "bold red"
            rec_icon = "💸"
        elif "WAIT" in result.recommendation.upper():
            rec_color = "yellow"
            rec_icon = "🍿"

        rec_panel = Panel(
            Align.center(Text(f"{rec_icon} {result.recommendation}", style=rec_color)),
            title="⚡ Action Plan",
            border_style="blue",
        )

        # 5. Narrative Text (re-using logic but formatted)
        narrative_text = Text()
        if "Bull" in result.trend:
            narrative_text.append("🐂 Bullish Vibes Detected! ", style="green")
            narrative_text.append("Buyers are in control. ", style="dim white")
        elif "Bear" in result.trend:
            narrative_text.append("🐻 Bearish Grip! ", style="red")
            narrative_text.append("Sellers are dominating. ", style="dim white")
        else:
            narrative_text.append("🦀 Crab Market! ", style="yellow")
            narrative_text.append("Sideways chop, be careful. ", style="dim white")

        if result.rsi > 70:
            narrative_text.append(
                "\n⚠️ Warning: Market is overheated (Overbought). Expect turbulence! ",
                style="bold red",
            )
        elif result.rsi < 30:
            narrative_text.append(
                "\n💎 Opportunity: Market is on sale (Oversold). Bargain hunting time? ",
                style="bold green",
            )

        nearest_s = (
            max([s.level for s in result.support_levels])
            if result.support_levels
            else 0
        )
        nearest_r = (
            min([r.level for r in result.resistance_levels])
            if result.resistance_levels
            else 0
        )

        if nearest_s and nearest_r:
            risk = result.current_price - nearest_s
            reward = nearest_r - result.current_price
            if risk > 0:
                rr_ratio = reward / risk
                narrative_text.append(
                    f"\n🎲 Risk/Reward Ratio: 1:{rr_ratio:.1f}", style="cyan"
                )

        # Assemble the full report
        main_group = Group(
            Panel(header_grid, style="blue"),
            metrics_table,
            levels_table,
            narrative_text,
            rec_panel,
        )

        return main_group

    def generate_chat_report(self, result: AnalysisResult) -> str:
        """Generate a compact, emoji-rich summary for chat apps (Telegram/WhatsApp)"""

        # 1. Determine Icons
        trend_icon = (
            "🚀" if "Bull" in result.trend else "🐻" if "Bear" in result.trend else "💤"
        )
        if "Golden Cross" in result.trend:
            trend_icon = "✨🚀"
        if "Death Cross" in result.trend:
            trend_icon = "☠️🐻"

        change_icon = "🟢" if result.change_percent >= 0 else "🔴"

        # 2. Key Levels
        s_level = (
            max([s.level for s in result.support_levels])
            if result.support_levels
            else 0
        )
        r_level = (
            min([r.level for r in result.resistance_levels])
            if result.resistance_levels
            else 0
        )

        # 3. Action Signal
        action = "WAIT"
        if "BUY" in result.recommendation.upper():
            action = "BUY 🟢"
        elif "SELL" in result.recommendation.upper():
            action = "SELL 🔴"
        elif "HOLD" in result.recommendation.upper():
            action = "HOLD ✊"

        # 4. Construct Message
        lines = [
            f"📊 *{result.ticker} Daily Update*",
            f"{change_icon} Price: {result.current_price:,.0f} ({result.change_percent:+.2f}%)",
            f"🌊 Trend: {trend_icon} {result.trend.split(' (')[0]}",
            "",
            f"📉 *Tech Stats:*",
            f"• RSI: {result.rsi:.1f}",
            f"• Vol: {result.volume / 1e6:.1f}M",
            "",
            f"🎯 *Key Levels:*",
            f"• 🧱 Res: {r_level:,.0f}" if r_level else "• 🧱 Res: -",
            f"• 🛡️ Sup: {s_level:,.0f}" if s_level else "• 🛡️ Sup: -",
            "",
            f"💡 *Outlook:*",
            f"{result.recommendation}",
            "",
            f"🚨 *Action:* {action}",
        ]

        return "\n".join(lines)

    def _strip_emojis(self, text: str) -> str:
        """Remove emojis from text for matplotlib compatibility"""
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"  # enclosed characters
            "\U0001f900-\U0001f9ff"  # supplemental symbols (🧊🥀🧱)
            "\U0001fa00-\U0001fa6f"  # chess symbols
            "\U0001fa70-\U0001faff"  # symbols and pictographs extended-a
            "\U00002600-\U000026ff"  # miscellaneous symbols
            "\U00002700-\U000027bf"  # dingbats
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r"", text).strip()

    def generate_chart(
        self, output_path: Optional[str] = None, show: bool = False
    ) -> str:
        """Generate technical analysis chart"""
        if self.hist is None:
            raise ChartError(
                "No data available",
                "Call fetch_data() before generate_chart()",
            )

        import os

        try:
            result = self.analyze()

            if output_path is None:
                output_path = f"{self.ticker.replace('.JK', '')}_chart.png"

            cfg = self.config.chart
            # Wider layout for dashboard style
            fig = plt.figure(
                figsize=(cfg.width * 1.2, cfg.height), layout="constrained"
            )

            # Dashboard Grid: 3 Rows, 2 Columns
            # Rows: Price (3), Volume (1), RSI (1)
            # Cols: Main Charts (4), Side Panel (1)
            gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], width_ratios=[4, 1])

            # --- Main Price Chart (Top Left) ---
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title(
                f"{result.ticker} Technical Analysis",
                fontsize=16,
                fontweight="bold",
            )
            ax1.plot(
                self.hist.index,
                self.hist["Close"],
                label="Price",
                color="#2E86AB",
                linewidth=2,
            )
            ax1.axhline(y=result.current_price, color="black", linestyle="-", alpha=0.5)

            sma_20_series = self._calculate_sma(20)
            sma_50_series = self._calculate_sma(50)
            sma_200_series = None
            if len(self.hist) >= 200:
                sma_200_series = self._calculate_sma(200)

            ax1.plot(
                self.hist.index,
                sma_20_series,
                label="SMA 20",
                color="#F59E0B",
                linewidth=1.5,
                alpha=0.8,
            )
            ax1.plot(
                self.hist.index,
                sma_50_series,
                label="SMA 50",
                color="#8B5CF6",
                linewidth=1.5,
                alpha=0.8,
            )
            if sma_200_series is not None:
                ax1.plot(
                    self.hist.index,
                    sma_200_series,
                    label="SMA 200",
                    color="#EF4444",
                    linewidth=1.5,
                    alpha=0.8,
                )

            if len(self.hist) >= 20:
                bb_middle_series, bb_upper_series, bb_lower_series = (
                    self._calculate_bollinger_bands()
                )
                ax1.plot(
                    self.hist.index,
                    bb_upper_series,
                    label="BB Upper",
                    color="#94A3B8",
                    linewidth=1.0,
                    alpha=0.6,
                    linestyle="--",
                )
                ax1.plot(
                    self.hist.index,
                    bb_lower_series,
                    label="BB Lower",
                    color="#94A3B8",
                    linewidth=1.0,
                    alpha=0.6,
                    linestyle="--",
                )
                ax1.fill_between(
                    self.hist.index,
                    bb_upper_series,
                    bb_lower_series,
                    alpha=0.05,
                    color="#94A3B8",
                    label="BB Band",
                )

            ax1.legend(loc="upper left", fontsize=9)

            # --- Buy/Sell Zones on Price Chart ---
            if result.support_levels and result.resistance_levels:
                strong_supports = [
                    s.level
                    for s in result.support_levels
                    if s.strength in ["strong", "moderate"]
                ]
                if strong_supports:
                    buy_zone_top = result.current_price
                    buy_zone_bottom = max(strong_supports)
                    if buy_zone_bottom < buy_zone_top:
                        ax1.axhspan(
                            buy_zone_bottom,
                            buy_zone_top,
                            alpha=0.1,
                            color="green",
                            label="Buy Zone",
                        )

                strong_resistances = [
                    r.level
                    for r in result.resistance_levels
                    if r.strength in ["strong", "moderate"]
                ]
                if strong_resistances:
                    sell_zone_bottom = result.current_price
                    sell_zone_top = min(strong_resistances)
                    if sell_zone_top > sell_zone_bottom:
                        ax1.axhspan(
                            sell_zone_bottom,
                            sell_zone_top,
                            alpha=0.1,
                            color="red",
                            label="Target Zone",
                        )

            for s in result.support_levels[:2]:
                ax1.axhline(y=s.level, color="green", linestyle="--", alpha=0.6)
                ax1.text(
                    self.hist.index[-1],
                    s.level,
                    f" S: {s.level:,.0f}",
                    color="green",
                    fontweight="bold",
                )

            for r in result.resistance_levels[:2]:
                ax1.axhline(y=r.level, color="red", linestyle="--", alpha=0.6)
                ax1.text(
                    self.hist.index[-1],
                    r.level,
                    f" R: {r.level:,.0f}",
                    color="red",
                    fontweight="bold",
                )

            # --- Volume Profile (Top Right) ---
            if result.vp_poc is not None and len(self.hist) >= 20:
                ax_vp = fig.add_subplot(gs[0, 1], sharey=ax1)

                price_low = float(self.hist["Low"].min())
                price_high = float(self.hist["High"].max())
                num_bins = 50
                bins = np.linspace(price_low, price_high, num_bins + 1)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                volume_by_bin = np.zeros(num_bins)

                for i in range(len(self.hist)):
                    row = self.hist.iloc[i]
                    candle_low = float(row["Low"])
                    candle_high = float(row["High"])
                    candle_volume = float(row["Volume"])

                    low_bin_idx = np.searchsorted(bins, candle_low, side="left") - 1
                    high_bin_idx = np.searchsorted(bins, candle_high, side="right") - 1

                    low_bin_idx = max(0, min(low_bin_idx, num_bins - 1))
                    high_bin_idx = max(0, min(high_bin_idx, num_bins - 1))

                    if low_bin_idx == high_bin_idx:
                        volume_by_bin[low_bin_idx] += candle_volume
                    else:
                        bins_in_range = high_bin_idx - low_bin_idx + 1
                        volume_per_bin = candle_volume / bins_in_range
                        for j in range(low_bin_idx, high_bin_idx + 1):
                            volume_by_bin[j] += volume_per_bin

                max_vol = volume_by_bin.max()
                colors = [
                    "#3B82F6"
                    if abs(bc - result.vp_poc) < (price_high - price_low) / num_bins * 2
                    else "#94A3B8"
                    for bc in bin_centers
                ]

                ax_vp.barh(
                    bin_centers,
                    volume_by_bin,
                    height=(price_high - price_low) / num_bins * 0.8,
                    color=colors,
                    alpha=0.7,
                )
                ax_vp.axhline(
                    y=result.vp_poc,
                    color="#EF4444",
                    linewidth=2,
                    linestyle="-",
                    label=f"POC: {result.vp_poc:,.0f}",
                )

                if result.vp_value_area_high and result.vp_value_area_low:
                    ax_vp.axhspan(
                        result.vp_value_area_low,
                        result.vp_value_area_high,
                        alpha=0.1,
                        color="blue",
                        label="Value Area",
                    )

                ax_vp.set_xlabel("Volume", fontsize=9)
                ax_vp.set_title("Volume Profile", fontsize=10, fontweight="bold")
                # ax_vp.legend(loc="upper right", fontsize=8) # Too crowded
                ax_vp.tick_params(axis="y", labelleft=False)
                ax_vp.set_xlim(0, max_vol * 1.2)
            else:
                # Placeholder if no VP data
                ax_vp = fig.add_subplot(gs[0, 1])
                ax_vp.axis("off")

            # --- Volume Bars (Middle Left) ---
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2.bar(self.hist.index, self.hist["Volume"], color="gray", alpha=0.5)
            ax2.set_ylabel("Volume", fontsize=9)
            ax2.grid(True, alpha=0.3)

            # --- RSI (Bottom Left) ---
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
            rsi = self._calculate_rsi()
            ax3.plot(self.hist.index, rsi, color="#8B5CF6", label="RSI(14)")
            ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
            ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
            ax3.set_ylim(0, 100)
            ax3.set_ylabel("RSI", fontsize=9)

            ax3.axhspan(70, 100, alpha=0.1, color="red")
            ax3.axhspan(0, 30, alpha=0.1, color="green")

            # Hide x-labels for top plots to clean up
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)

            # --- Analysis Card (Bottom Right - spanning 2 rows) ---
            ax_insight = fig.add_subplot(gs[1:, 1])
            ax_insight.axis("off")

            trend_color = (
                "#22c55e"
                if "Bull" in result.trend
                else "#ef4444"
                if "Bear" in result.trend
                else "#f59e0b"
            )
            trend_emoji = (
                "BULL"
                if "Bull" in result.trend
                else "BEAR"
                if "Bear" in result.trend
                else "NEUTRAL"
            )

            nearest_support = (
                max([s.level for s in result.support_levels])
                if result.support_levels
                else result.current_price * 0.95
            )
            nearest_resistance = (
                min([r.level for r in result.resistance_levels])
                if result.resistance_levels
                else result.current_price * 1.05
            )
            risk_reward = (
                (nearest_resistance - result.current_price)
                / (result.current_price - nearest_support)
                if nearest_support < result.current_price
                else 0
            )

            narrative = self._generate_narrative(
                result, trend_emoji, nearest_support, nearest_resistance, risk_reward
            )

            # Format narrative for the side panel
            # Breaking lines manually for the narrow column
            formatted_narrative = narrative.replace(" | ", "\n").replace(". ", ".\n")

            # Add box
            ax_insight.text(
                0.5,
                0.5,
                self._strip_emojis(formatted_narrative),
                transform=ax_insight.transAxes,
                fontsize=9,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="#f8fafc",
                    edgecolor=trend_color,
                    linewidth=2,
                    alpha=0.95,
                ),
                wrap=True,
            )

            # Footer
            fig.text(
                0.5,
                0.01,
                f"Analysis generated by IDX Analyzer on {time.strftime('%Y-%m-%d')}",
                ha="center",
                fontsize=8,
                color="gray",
                style="italic",
            )

            plt.savefig(output_path, dpi=cfg.dpi)
            if show:
                plt.show()
            else:
                plt.close()

            return os.path.abspath(output_path)

        except ChartError:
            raise
        except Exception as e:
            raise ChartError(
                "Failed to generate chart",
                str(e),
            )
