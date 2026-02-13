"""
Core analysis module for IDX stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import os


@dataclass
class SupportResistance:
    """Support and resistance levels"""

    level: float
    type: str  # 'support' or 'resistance'
    strength: str  # 'weak', 'moderate', 'strong'
    description: str


@dataclass
class AnalysisResult:
    """Complete analysis result"""

    ticker: str
    current_price: float
    change_percent: float
    volume: int
    week_52_high: float
    week_52_low: float
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]
    trend: str
    recommendation: str
    summary: str
    rsi: float
    sma_20: float
    sma_50: float
    sma_200: Optional[float]
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None


class IDXAnalyzer:
    """Analyzer for Indonesian stocks (IDX)"""

    def __init__(self, ticker: str):
        """
        Initialize analyzer with ticker

        Args:
            ticker: Stock ticker (e.g., 'BBCA.JK', 'TLKM.JK')
        """
        self.ticker = self._format_ticker(ticker)
        self.stock = yf.Ticker(self.ticker)
        self.hist = None
        self.info = None

    def _format_ticker(self, ticker: str) -> str:
        """Format ticker to Yahoo Finance format"""
        # Remove IDX: prefix if present
        if ticker.upper().startswith("IDX:"):
            ticker = ticker[4:]

        # Add .JK suffix if not present
        if not ticker.endswith(".JK"):
            ticker = f"{ticker}.JK"

        return ticker.upper()

    def fetch_data(self, period: str = "6mo") -> bool:
        """
        Fetch historical data and stock info
        """
        try:
            self.hist = self.stock.history(period=period)

            # Use faster metadata fetching if possible
            try:
                self.info = self.stock.info
            except:
                self.info = {}

            return len(self.hist) > 0
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False

    def _calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        if self.hist is None or len(self.hist) < window:
            return pd.Series([50.0] * len(self.hist))

        delta = self.hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_sma(self, window: int) -> pd.Series:
        if self.hist is None or len(self.hist) < window:
            return (
                pd.Series([self.hist["Close"].iloc[-1]] * len(self.hist))
                if self.hist is not None
                else pd.Series([])
            )
        return self.hist["Close"].rolling(window=window).mean()

    def _find_support_levels(self) -> List[SupportResistance]:
        """Find key support levels"""
        supports = []
        if self.hist is None or len(self.hist) == 0:
            return supports

        low_52w = self.hist["Low"].min()
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
        for i, low in enumerate(recent_lows):
            if low > low_52w * 1.02:
                supports.append(
                    SupportResistance(
                        level=round(low, 0),
                        type="support",
                        strength="moderate",
                        description=f"Recent swing low",
                    )
                )

        current = self.hist["Close"].iloc[-1]
        round_levels = [int(current / 100) * 100, int(current / 500) * 500]
        for level in set(round_levels):
            if level < current and level > low_52w:
                supports.append(
                    SupportResistance(
                        level=level,
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

        current = self.hist["Close"].iloc[-1]
        high_52w = self.hist["High"].max()
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
        for i, high in enumerate(recent_highs):
            if high > current * 1.01:
                resistances.append(
                    SupportResistance(
                        level=round(high, 0),
                        type="resistance",
                        strength="moderate",
                        description=f"Recent swing high",
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
                        level=level,
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
        if self.hist is None or len(self.hist) < 20:
            return "Unknown"

        current = self.hist["Close"].iloc[-1]

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
            if "Bullish" in trend and sma_20 > sma_50:
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

        current = self.hist["Close"].iloc[-1]
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
        if self.hist is None:
            raise ValueError("Call fetch_data() first")

        current = self.hist["Close"].iloc[-1]
        prev_close = self.hist["Close"].iloc[-2] if len(self.hist) > 1 else current
        change_pct = (current - prev_close) / prev_close * 100

        rsi_series = self._calculate_rsi()
        current_rsi = rsi_series.iloc[-1]

        sma_20_series = self._calculate_sma(20)
        sma_50_series = self._calculate_sma(50)
        sma_200_series = self._calculate_sma(200) if len(self.hist) >= 200 else None

        sma_20 = float(sma_20_series.iloc[-1]) if len(sma_20_series) > 0 else current
        sma_50 = float(sma_50_series.iloc[-1]) if len(sma_50_series) > 0 else current
        sma_200 = (
            float(sma_200_series.iloc[-1])
            if sma_200_series is not None and len(sma_200_series) > 0
            else None
        )

        supports = self._find_support_levels()
        resistances = self._find_resistance_levels()
        trend = self._determine_trend(current_rsi, sma_20, sma_50, sma_200)
        recommendation = self._generate_recommendation(
            trend, current_rsi, supports, resistances
        )

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
            market_cap=self.info.get("marketCap") if self.info else None,
            pe_ratio=self.info.get("trailingPE") if self.info else None,
            dividend_yield=self.info.get("dividendYield") if self.info else None,
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
        lines = [
            f"💰 Price: {current:,.0f} | RSI: {rsi:.1f} | Trend: {trend}",
            f"🏛️ Fundamentals: MC: {self.info.get('marketCap', 0) / 1e12:.1f}T | P/E: {self.info.get('trailingPE', 0):.1f}",
        ]
        return "\n".join(lines)

    def generate_chart(self, output_path: str = None, show: bool = False) -> str:
        """Generate technical analysis chart"""
        if self.hist is None:
            raise ValueError("Call fetch_data() first")
        if output_path is None:
            output_path = f"{self.ticker.replace('.JK', '')}_chart.png"

        result = self.analyze()
        fig = plt.figure(figsize=(14, 10), layout="constrained")
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax1.set_title(
            f"{result.ticker} Technical Analysis", fontsize=16, fontweight="bold"
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
        sma_200_series = self._calculate_sma(200) if len(self.hist) >= 200 else None

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

        ax1.legend(loc="upper left", fontsize=9)

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

        insight_text = (
            f"[{trend_emoji}] TREND: {result.trend}\n"
            f"Price: {result.current_price:,.0f}\n"
            f"RSI: {result.rsi:.1f} ({'Overbought' if result.rsi > 70 else 'Oversold' if result.rsi < 30 else 'Neutral'})\n"
            f"Target: {nearest_resistance:,.0f} (+{((nearest_resistance / result.current_price - 1) * 100):.1f}%)\n"
            f"Support: {nearest_support:,.0f} ({((nearest_support / result.current_price - 1) * 100):.1f}%)\n"
        )

        if risk_reward > 0:
            insight_text += f"R/R Ratio: 1:{risk_reward:.1f}\n"

        if "Golden Cross" in result.trend:
            insight_text += "\n[STRONG BUY]\nSMA 50 > 200\nUptrend Confirmed"
        elif "Death Cross" in result.trend:
            insight_text += "\n[STRONG SELL]\nSMA 50 < 200\nDowntrend Active"
        elif "Bull" in result.trend and result.rsi < 70:
            insight_text += "\n[BUY ZONE]\nAbove key MAs\nMomentum positive"
        elif "Bear" in result.trend and result.rsi > 30:
            insight_text += "\n[SELL ZONE]\nBelow key MAs\nMomentum negative"
        elif result.rsi > 70:
            insight_text += "\n[OVERBOUGHT]\nConsider taking profits"
        elif result.rsi < 30:
            insight_text += "\n[OVERSOLD]\nPotential bounce coming"
        else:
            insight_text += "\n[WAIT]\nNo clear signal"

        ax1.text(
            0.98,
            0.98,
            insight_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=trend_color,
                linewidth=2,
                alpha=0.95,
            ),
            family="monospace",
        )

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
                    ax1.text(
                        self.hist.index[int(len(self.hist) * 0.02)],
                        (buy_zone_top + buy_zone_bottom) / 2,
                        " BUY",
                        color="green",
                        fontsize=8,
                        va="center",
                        alpha=0.7,
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
                    ax1.text(
                        self.hist.index[int(len(self.hist) * 0.02)],
                        (sell_zone_top + sell_zone_bottom) / 2,
                        " TARGET",
                        color="red",
                        fontsize=8,
                        va="center",
                        alpha=0.7,
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

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(self.hist.index, self.hist["Volume"], color="gray", alpha=0.5)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        rsi = self._calculate_rsi()
        ax3.plot(self.hist.index, rsi, color="#8B5CF6", label="RSI(14)")
        ax3.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        ax3.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI", fontsize=9)

        ax3.axhspan(70, 100, alpha=0.1, color="red")
        ax3.axhspan(0, 30, alpha=0.1, color="green")

        ax3.text(
            self.hist.index[int(len(self.hist) * 0.02)],
            85,
            "OVERBOUGHT (SELL)",
            fontsize=8,
            color="red",
            alpha=0.7,
            fontweight="bold",
        )
        ax3.text(
            self.hist.index[int(len(self.hist) * 0.02)],
            15,
            "OVERSOLD (BUY)",
            fontsize=8,
            color="green",
            alpha=0.7,
            fontweight="bold",
        )

        info_text = f"Price: {result.current_price:,.0f} | RSI: {result.rsi:.1f} | {result.recommendation}"
        fig.text(
            0.5,
            0.02,
            info_text,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.savefig(output_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
        return os.path.abspath(output_path)
