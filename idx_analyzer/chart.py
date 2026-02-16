"""
Unified chart generation module for IDX Analyzer.

Supports two chart styles:
- 'standard': Classic technical analysis chart with Catppuccin Mocha theme
- 'executive': High-end executive dashboard with Tailwind CSS design principles

Usage:
    from .chart import generate_chart

    # Standard chart
    generate_chart(analyzer, style='standard', output_path='chart.png')

    # Executive dashboard
    generate_chart(analyzer, style='executive', output_path='executive.png')
"""

from typing import TYPE_CHECKING, Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from .analyzer import IDXAnalyzer


# ============================================================================
# UNIFIED THEME SYSTEM
# ============================================================================
class ChartTheme:
    """Unified color theme for both chart styles."""

    # Catppuccin Mocha Palette (Standard Charts)
    CTP = {
        "bg_facecolor": "#1e1e2e",  # Base
        "axes_facecolor": "#313244",  # Surface0
        "surface0": "#45475a",  # Surface1
        "text": "#cdd6f4",  # Text
        "title_color": "#f5e0dc",  # Rosewater
        "price_line": "#89b4fa",  # Blue
        "current_price_line": "#a6e3a1",  # Green
        "sma_20": "#fab387",  # Peach
        "sma_50": "#cba6f7",  # Mauve
        "sma_200": "#94e2d5",  # Teal
        "bb_upper": "#f5c2e7",  # Pink
        "bb_lower": "#f5c2e7",
        "bb_fill": "#f5c2e7",
        "support": "#a6e3a1",  # Green
        "resistance": "#f38ba8",  # Red
        "support_zone": "#a6e3a1",
        "resistance_zone": "#f38ba8",
        "volume_up": "#a6e3a1",
        "volume_down": "#f38ba8",
        "rsi_line": "#f9e2af",  # Yellow
        "rsi_overbought": "#f38ba8",
        "rsi_oversold": "#a6e3a1",
        "tick_color": "#cdd6f4",
        "grid_color": "#6c7086",
        "section_header": "#f5e0dc",
        "peer_up": "#a6e3a1",
        "peer_down": "#f38ba8",
        "peer_neutral": "#9399b2",
    }

    ICONS = {
        "bull": "▲",
        "bear": "▼",
        "neutral": "◆",
        "up": "▲",
        "down": "▼",
        "support": "🛡️",
        "resistance": "🧱",
        "sector": "◐",
    }


class ExecutiveTheme:
    """Tailwind-inspired theme for executive dashboard."""

    # Background Colors
    BG_PRIMARY = "#1a1b26"  # Dark background
    BG_SECONDARY = "#1f2335"  # Card background

    # Text Colors
    TEXT_PRIMARY = "#c0caf5"  # Main text
    TEXT_SECONDARY = "#565f89"  # Labels, muted text
    TEXT_MUTED = "#414868"  # Grid lines
    TEXT_WHITE = "#ffffff"  # Emphasis

    # Accent Colors (Catppuccin Mocha)
    BULLISH = "#4ade80"  # Green
    BEARISH = "#f87171"  # Red
    NEUTRAL = "#fbbf24"  # Yellow/Amber
    WARNING = "#fbbf24"

    # Chart Lines
    PRICE_LINE = "#7aa2f7"  # Blue
    SMA_20 = "#ff9e64"  # Orange
    SMA_50 = "#bb9af7"  # Purple
    BBANDS = "#7dcfff"  # Cyan
    RSI_LINE = "#ff9e64"  # Orange

    # Border
    BORDER = "#24283b"

    @classmethod
    def get_action_style(cls, action: str) -> dict:
        """Get style dict for action badge."""
        action_upper = action.upper()
        if "STRONG BUY" in action_upper:
            return {"color": "#22c55e", "bgcolor": "#0f3d1f", "border": "#15803d"}
        elif "BUY" in action_upper:
            return {"color": cls.BULLISH, "bgcolor": "#1a3a2a", "border": "#2d5a3d"}
        elif "STRONG SELL" in action_upper:
            return {"color": "#dc2626", "bgcolor": "#3d0f0f", "border": "#991b1b"}
        elif "SELL" in action_upper:
            return {"color": cls.BEARISH, "bgcolor": "#3a1a1a", "border": "#5a2d2d"}
        else:
            return {"color": cls.NEUTRAL, "bgcolor": "#1a233a", "border": "#2d3d5a"}


# ============================================================================
# TYPOGRAPHY & SPACING (Tailwind-inspired)
# ============================================================================
class Typography:
    XS = 9
    SM = 10
    BASE = 11
    LG = 12
    XL = 14
    XL2 = 16
    XL3 = 18
    XL4 = 22


class Spacing:
    XS = 4
    SM = 8
    MD = 16
    LG = 24
    XL = 32


# ============================================================================
# STANDARD CHART GENERATION
# ============================================================================
def generate_standard_chart(
    analyzer: "IDXAnalyzer",
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """Generate standard technical analysis chart with Catppuccin theme."""
    from .analyzer import ChartError

    if analyzer.hist is None:
        raise ChartError(
            "No data available", "Call fetch_data() before generate_chart()"
        )

    result = analyzer.analyze()
    hist = analyzer.hist
    CTP = ChartTheme.CTP
    ICONS = ChartTheme.ICONS

    fig = plt.figure(
        figsize=(14, 10),
        facecolor=CTP["bg_facecolor"],
        layout="constrained",
    )

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[3, 1, 1],
        width_ratios=[4, 1],
        hspace=0.05,
        wspace=0.05,
    )

    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.1)

    # === PRICE ACTION ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CTP["axes_facecolor"])

    trend_icon = (
        ICONS["bull"]
        if "Bull" in result.trend
        else ICONS["bear"]
        if "Bear" in result.trend
        else ICONS["neutral"]
    )
    ax1.set_title(
        f"{trend_icon} {result.ticker} Price Action",
        fontsize=16,
        fontweight="bold",
        color=CTP["title_color"],
        pad=10,
    )

    ax1.plot(
        hist.index, hist["Close"], label="Price", color=CTP["price_line"], linewidth=2
    )
    ax1.axhline(
        y=result.current_price,
        color=CTP["current_price_line"],
        linestyle="-",
        alpha=0.5,
    )

    # Moving averages
    sma_20 = hist["Close"].rolling(20).mean()
    sma_50 = hist["Close"].rolling(50).mean()
    ax1.plot(hist.index, sma_20, label="SMA 20", color=CTP["sma_20"], linewidth=1.5)
    ax1.plot(hist.index, sma_50, label="SMA 50", color=CTP["sma_50"], linewidth=1.5)

    if len(hist) >= 200:
        sma_200 = hist["Close"].rolling(200).mean()
        ax1.plot(
            hist.index, sma_200, label="SMA 200", color=CTP["sma_200"], linewidth=1.5
        )

    # Bollinger Bands
    if len(hist) >= 20:
        bb_std = hist["Close"].rolling(20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        ax1.plot(
            hist.index,
            bb_upper,
            label="BB Upper",
            color=CTP["bb_upper"],
            linewidth=1,
            alpha=0.6,
            linestyle="--",
        )
        ax1.plot(
            hist.index,
            bb_lower,
            label="BB Lower",
            color=CTP["bb_lower"],
            linewidth=1,
            alpha=0.6,
            linestyle="--",
        )
        ax1.fill_between(
            hist.index, bb_upper, bb_lower, alpha=0.1, color=CTP["bb_fill"]
        )

    ax1.legend(
        loc="upper left",
        fontsize=9,
        facecolor=CTP["axes_facecolor"],
        edgecolor=CTP["surface0"],
        labelcolor=CTP["text"],
    )

    # Support/Resistance lines
    for s in result.support_levels[:2]:
        ax1.axhline(y=s.level, color=CTP["support"], linestyle="--", alpha=0.6)
        ax1.text(
            hist.index[-1],
            s.level,
            f" S {s.level:,.0f}",
            color=CTP["support"],
            fontweight="bold",
            fontsize=9,
        )

    for r in result.resistance_levels[:2]:
        ax1.axhline(y=r.level, color=CTP["resistance"], linestyle="--", alpha=0.6)
        ax1.text(
            hist.index[-1],
            r.level,
            f" R {r.level:,.0f}",
            color=CTP["resistance"],
            fontweight="bold",
            fontsize=9,
        )

    ax1.tick_params(colors=CTP["tick_color"])
    ax1.grid(True, alpha=0.2, color=CTP["grid_color"])
    for spine in ax1.spines.values():
        spine.set_color(CTP["surface0"])

    # === SECTOR PULSE ===
    try:
        from .sector_comparison import get_peer_table_data

        peer_data = get_peer_table_data(result.ticker, max_peers=3)

        ax_peers = fig.add_subplot(gs_right[0, 0])
        ax_peers.set_facecolor(CTP["axes_facecolor"])
        ax_peers.set_xlim(0, 1)
        ax_peers.set_ylim(0, 1)
        ax_peers.set_xticks([])
        ax_peers.set_yticks([])
        for spine in ax_peers.spines.values():
            spine.set_color(CTP["surface0"])

        if peer_data["peers"]:
            ax_peers.text(
                0.5,
                0.92,
                f"{ICONS['sector']} {peer_data['sector'].upper()}",
                transform=ax_peers.transAxes,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="top",
                color=CTP["section_header"],
            )

            y_pos = 0.65
            for peer in peer_data["peers"][:3]:
                icon = ICONS["up"] if peer.is_up else ICONS["down"]
                color = CTP["peer_up"] if peer.is_up else CTP["peer_down"]
                ax_peers.text(
                    0.5,
                    y_pos,
                    f"{icon} {peer.ticker} {peer.price:,.0f} ({peer.change_percent:+.1f}%)",
                    transform=ax_peers.transAxes,
                    fontsize=10,
                    ha="center",
                    va="top",
                    color=color,
                    fontweight="bold",
                )
                y_pos -= 0.28
    except Exception:
        ax_peers = fig.add_subplot(gs_right[0, 0])
        ax_peers.set_facecolor(CTP["axes_facecolor"])
        ax_peers.axis("off")

    # === VOLUME PROFILE ===
    if len(hist) >= 20:
        ax_vp = fig.add_subplot(gs_right[1, 0], sharey=ax1)
        ax_vp.set_facecolor(CTP["axes_facecolor"])

        price_low = float(hist["Low"].min())
        price_high = float(hist["High"].max())
        num_bins = 50
        bins = np.linspace(price_low, price_high, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_by_bin = np.zeros(num_bins)

        for i in range(len(hist)):
            row = hist.iloc[i]
            candle_low, candle_high, candle_volume = (
                float(row["Low"]),
                float(row["High"]),
                float(row["Volume"]),
            )
            low_idx = max(
                0, min(np.searchsorted(bins, candle_low, side="left") - 1, num_bins - 1)
            )
            high_idx = max(
                0,
                min(np.searchsorted(bins, candle_high, side="right") - 1, num_bins - 1),
            )
            if low_idx == high_idx:
                volume_by_bin[low_idx] += candle_volume
            else:
                vol_per = candle_volume / (high_idx - low_idx + 1)
                for j in range(low_idx, high_idx + 1):
                    volume_by_bin[j] += vol_per

        max_vol = volume_by_bin.max()
        ax_vp.barh(
            bin_centers,
            volume_by_bin,
            height=(price_high - price_low) / num_bins * 0.8,
            color=CTP["price_line"],
            alpha=0.6,
        )
        ax_vp.set_xlim(0, max_vol * 1.1)
        ax_vp.axis("off")

    # === VOLUME CHART ===
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.set_facecolor(CTP["axes_facecolor"])

    colors = [
        CTP["volume_up"]
        if hist["Close"].iloc[i] >= hist["Open"].iloc[i]
        else CTP["volume_down"]
        for i in range(len(hist))
    ]
    ax2.bar(hist.index, hist["Volume"], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume", color=CTP["text"])
    ax2.tick_params(colors=CTP["tick_color"])
    ax2.grid(True, alpha=0.2, color=CTP["grid_color"], axis="y")
    for spine in ax2.spines.values():
        spine.set_color(CTP["surface0"])

    # === RSI CHART ===
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.set_facecolor(CTP["axes_facecolor"])

    delta = hist["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    ax3.plot(hist.index, rsi, color=CTP["rsi_line"], linewidth=1.5)
    ax3.axhline(y=70, color=CTP["rsi_overbought"], linestyle="--", alpha=0.5)
    ax3.axhline(y=30, color=CTP["rsi_oversold"], linestyle="--", alpha=0.5)
    ax3.fill_between(hist.index, 70, 100, alpha=0.1, color=CTP["rsi_overbought"])
    ax3.fill_between(hist.index, 0, 30, alpha=0.1, color=CTP["rsi_oversold"])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color=CTP["text"])
    ax3.tick_params(colors=CTP["tick_color"])
    ax3.grid(True, alpha=0.2, color=CTP["grid_color"], axis="y")
    for spine in ax3.spines.values():
        spine.set_color(CTP["surface0"])

    # Save
    if output_path is None:
        output_path = f"{result.ticker}_chart.png"

    plt.savefig(
        output_path,
        dpi=150,
        facecolor=CTP["bg_facecolor"],
        edgecolor="none",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


# ============================================================================
# EXECUTIVE DASHBOARD GENERATION
# ============================================================================
def _add_card_container(ax, padding: float = 0.015, radius: float = 0.010) -> None:
    """Add card container with rounded corners."""
    rect = mpatches.FancyBboxPatch(
        (padding, padding),
        1 - 2 * padding,
        1 - 2 * padding,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=ExecutiveTheme.BG_SECONDARY,
        edgecolor=ExecutiveTheme.BORDER,
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
        zorder=0,
    )
    ax.add_patch(rect)


def _create_gauge_chart(ax, value: float, vmin: float = 0, vmax: float = 100) -> None:
    """Create semi-circle gauge for volume ratio."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.set_facecolor(ExecutiveTheme.BG_SECONDARY)
    ax.set_aspect("equal")

    # Draw zones
    theta_green = np.linspace(0, np.pi * 0.4, 50)
    theta_yellow = np.linspace(np.pi * 0.4, np.pi * 0.6, 50)
    theta_red = np.linspace(np.pi * 0.6, np.pi, 50)

    for theta, color in [
        (theta_green, ExecutiveTheme.BULLISH),
        (theta_yellow, ExecutiveTheme.WARNING),
        (theta_red, ExecutiveTheme.BEARISH),
    ]:
        for i in range(len(theta) - 1):
            x = [
                0.6 * np.cos(theta[i]),
                1.0 * np.cos(theta[i]),
                1.0 * np.cos(theta[i + 1]),
                0.6 * np.cos(theta[i + 1]),
            ]
            y = [
                0.6 * np.sin(theta[i]),
                1.0 * np.sin(theta[i]),
                1.0 * np.sin(theta[i + 1]),
                0.6 * np.sin(theta[i + 1]),
            ]
            ax.fill(x, y, color=color, alpha=0.3)

    # Outer arc
    theta_all = np.linspace(0, np.pi, 100)
    ax.plot(
        1.0 * np.cos(theta_all),
        1.0 * np.sin(theta_all),
        color=ExecutiveTheme.TEXT_SECONDARY,
        linewidth=2,
        alpha=0.5,
    )

    # Needle
    normalized = max(0, min(1, (value - vmin) / (vmax - vmin)))
    needle_angle = normalized * np.pi
    needle_x, needle_y = 0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)
    ax.annotate(
        "",
        xy=(needle_x, needle_y),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=ExecutiveTheme.TEXT_WHITE, lw=2.5),
    )

    # Center dot
    ax.add_patch(plt.Circle((0, 0), 0.08, color=ExecutiveTheme.TEXT_WHITE, zorder=10))

    # Value
    ax.text(
        0,
        -0.15,
        f"{value:.1f}",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        color=ExecutiveTheme.TEXT_WHITE,
    )

    # Label
    if value > 60:
        text, color = "Accumulation", ExecutiveTheme.BULLISH
    elif value < 40:
        text, color = "Distribution", ExecutiveTheme.BEARISH
    else:
        text, color = "Neutral", ExecutiveTheme.WARNING
    ax.text(
        0,
        -0.35,
        text,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color=color,
    )


def generate_executive_dashboard(
    analyzer: "IDXAnalyzer",
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """Generate high-end executive dashboard with Tailwind CSS design."""
    from .analyzer import ChartError

    if analyzer.hist is None:
        raise ChartError(
            "No data available", "Call fetch_data() before generate_chart()"
        )

    result = analyzer.analyze()
    hist = analyzer.hist
    T = ExecutiveTheme

    fig = plt.figure(figsize=(18, 14), facecolor=T.BG_PRIMARY)

    # Main grid: Title (8%), Metrics (12%), Content (80%)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.08, 0.12, 1],
        width_ratios=[2.6, 1],
        hspace=0.10,
        wspace=0.06,
        left=0.025,
        right=0.975,
        top=0.965,
        bottom=0.025,
    )

    # === ROW 1: TITLE ===
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")
    ax_title.set_facecolor(T.BG_PRIMARY)
    ax_title.text(
        0.5,
        0.45,
        f"{result.ticker} Executive Summary",
        fontsize=Typography.XL4,
        fontweight="bold",
        color=T.TEXT_WHITE,
        ha="center",
        va="center",
    )

    # === ROW 2: METRICS BAR ===
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.set_xlim(0, 100)
    ax_metrics.set_ylim(0, 10)
    ax_metrics.axis("off")

    # Background card
    rect = mpatches.FancyBboxPatch(
        (0.8, 0.8),
        98.4,
        8.4,
        boxstyle="round,pad=0.3,rounding_size=0.8",
        facecolor=T.BG_SECONDARY,
        edgecolor=T.BORDER,
        linewidth=1.5,
        transform=ax_metrics.transData,
    )
    ax_metrics.add_patch(rect)

    # Calculate metrics
    safety_net = (
        max([s.level for s in result.support_levels])
        if result.support_levels
        else result.current_price * 0.95
    )
    wall = (
        min([r.level for r in result.resistance_levels])
        if result.resistance_levels
        else result.current_price * 1.05
    )
    risk = result.current_price - safety_net
    reward = wall - result.current_price
    rr_ratio = reward / risk if risk > 0 else 0

    # Determine action
    action_text = "HOLD"
    if "Bull" in result.trend:
        action_text = "STRONG BUY" if result.rsi < 30 else "BUY"
    elif "Bear" in result.trend:
        action_text = "STRONG SELL" if result.rsi > 70 else "SELL"
    elif "BUY" in result.recommendation.upper():
        action_text = "BUY"
    elif (
        "SELL" in result.recommendation.upper()
        or "AVOID" in result.recommendation.upper()
    ):
        action_text = "SELL"

    action_style = T.get_action_style(action_text)

    # Metrics
    metrics = [
        (
            "Current Price",
            f"{result.current_price:,.0f}",
            T.BULLISH if result.change_percent >= 0 else T.BEARISH,
        ),
        ("Risk/Reward", f"1:{rr_ratio:.1f}", T.TEXT_WHITE),
        ("Safety Net", f"{safety_net:,.0f}", T.BULLISH),
        ("Wall", f"{wall:,.0f}", T.BEARISH),
    ]

    for (label, value, color), x in zip(metrics, [16, 34, 52, 70]):
        ax_metrics.text(
            x,
            6.2,
            label,
            fontsize=Typography.SM,
            color=T.TEXT_SECONDARY,
            ha="center",
            va="center",
        )
        ax_metrics.text(
            x,
            3.2,
            value,
            fontsize=Typography.XL2,
            color=color,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # Action badge
    action_rect = mpatches.FancyBboxPatch(
        (87, 2),
        12,
        6,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=action_style["bgcolor"],
        edgecolor=action_style["border"],
        linewidth=2,
        transform=ax_metrics.transData,
    )
    ax_metrics.add_patch(action_rect)
    ax_metrics.text(
        93,
        5,
        action_text,
        fontsize=Typography.XL3,
        fontweight="heavy",
        color=action_style["color"],
        ha="center",
        va="center",
    )

    # === ROW 3: CHARTS (60:20:20) ===
    gs_left = gs[2, 0].subgridspec(3, 1, hspace=0.05, height_ratios=[3.0, 1.0, 1.0])

    # Price Action (60%)
    ax_price = fig.add_subplot(gs_left[0, 0])
    ax_price.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_price)
    ax_price.text(
        0.015,
        0.97,
        "Price Action",
        fontsize=Typography.LG,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        transform=ax_price.transAxes,
        va="top",
        ha="left",
    )

    sma_20 = hist["Close"].rolling(20).mean()
    sma_50 = hist["Close"].rolling(50).mean()
    bb_std = hist["Close"].rolling(20).std()
    bb_upper, bb_lower = sma_20 + (bb_std * 2), sma_20 - (bb_std * 2)

    ax_price.plot(
        hist.index, hist["Close"], color=T.PRICE_LINE, linewidth=1.5, label="Price"
    )
    ax_price.plot(
        hist.index, sma_20, color=T.SMA_20, linewidth=1.2, alpha=0.8, label="SMA 20"
    )
    ax_price.plot(
        hist.index, sma_50, color=T.SMA_50, linewidth=1.2, alpha=0.8, label="SMA 50"
    )
    ax_price.plot(
        hist.index, bb_upper, color=T.BBANDS, linewidth=0.8, alpha=0.5, linestyle="--"
    )
    ax_price.plot(
        hist.index, bb_lower, color=T.BBANDS, linewidth=0.8, alpha=0.5, linestyle="--"
    )
    ax_price.fill_between(hist.index, bb_upper, bb_lower, alpha=0.08, color=T.BBANDS)
    ax_price.axhline(y=safety_net, color=T.BULLISH, linestyle="--", alpha=0.6)
    ax_price.axhline(y=wall, color=T.BEARISH, linestyle="--", alpha=0.6)
    ax_price.legend(
        loc="upper left",
        fontsize=Typography.XS,
        framealpha=0.95,
        facecolor=T.BG_SECONDARY,
        edgecolor=T.BORDER,
        labelcolor=T.TEXT_SECONDARY,
    )
    ax_price.tick_params(colors=T.TEXT_SECONDARY, labelsize=Typography.XS)
    ax_price.grid(True, alpha=0.12, color=T.TEXT_MUTED)
    for spine in ax_price.spines.values():
        spine.set_color(T.BORDER)
        spine.set_linewidth(0.5)
    ax_price.set_xticklabels([])

    # Volume Trend (20%)
    ax_volume = fig.add_subplot(gs_left[1, 0], sharex=ax_price)
    ax_volume.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_volume)
    ax_volume.text(
        0.015,
        0.97,
        "Volume Trend",
        fontsize=Typography.LG,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        transform=ax_volume.transAxes,
        va="top",
        ha="left",
    )

    colors = [
        T.BULLISH if hist["Close"].iloc[i] >= hist["Open"].iloc[i] else T.BEARISH
        for i in range(len(hist))
    ]
    ax_volume.bar(hist.index, hist["Volume"], color=colors, alpha=0.75, width=0.8)
    ax_volume.set_ylabel("Volume", color=T.TEXT_SECONDARY, fontsize=Typography.SM)
    ax_volume.tick_params(colors=T.TEXT_SECONDARY, labelsize=Typography.XS)
    ax_volume.grid(True, alpha=0.12, color=T.TEXT_MUTED, axis="y")
    for spine in ax_volume.spines.values():
        spine.set_color(T.BORDER)
        spine.set_linewidth(0.5)
    ax_volume.set_xticklabels([])

    # RSI Momentum (20%)
    ax_rsi = fig.add_subplot(gs_left[2, 0], sharex=ax_price)
    ax_rsi.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_rsi)
    ax_rsi.text(
        0.015,
        0.97,
        "RSI Momentum",
        fontsize=Typography.LG,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        transform=ax_rsi.transAxes,
        va="top",
        ha="left",
    )

    delta = hist["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    ax_rsi.plot(hist.index, rsi, color=T.RSI_LINE, linewidth=1.5)
    ax_rsi.axhline(y=70, color=T.BEARISH, linestyle="--", alpha=0.5)
    ax_rsi.axhline(y=30, color=T.BULLISH, linestyle="--", alpha=0.5)
    ax_rsi.fill_between(hist.index, 70, 100, alpha=0.1, color=T.BEARISH)
    ax_rsi.fill_between(hist.index, 0, 30, alpha=0.1, color=T.BULLISH)
    ax_rsi.text(
        hist.index[0], 72, "Overbought", color=T.BEARISH, fontsize=8, va="bottom"
    )
    ax_rsi.text(hist.index[0], 28, "Oversold", color=T.BULLISH, fontsize=8, va="top")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", color=T.TEXT_SECONDARY, fontsize=Typography.SM)
    ax_rsi.tick_params(colors=T.TEXT_SECONDARY, labelsize=Typography.XS)
    ax_rsi.grid(True, alpha=0.12, color=T.TEXT_MUTED, axis="y")
    for spine in ax_rsi.spines.values():
        spine.set_color(T.BORDER)
        spine.set_linewidth(0.5)

    # === RIGHT COLUMN: GAUGE, SECTOR, INSIGHTS ===
    gs_right = gs[2, 1].subgridspec(3, 1, hspace=0.08, height_ratios=[1.3, 1.0, 1.1])

    # Gauge
    ax_gauge = fig.add_subplot(gs_right[0, 0])
    ax_gauge.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_gauge)
    ax_gauge.text(
        0.5,
        0.94,
        "Accumulation vs Distribution",
        fontsize=Typography.BASE,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        ha="center",
        transform=ax_gauge.transAxes,
    )

    # Volume ratio calculation
    up_vol = hist[hist["Close"] > hist["Close"].shift(1)]["Volume"].sum()
    down_vol = hist[hist["Close"] < hist["Close"].shift(1)]["Volume"].sum()
    total_vol = up_vol + down_vol
    vp_ratio = (up_vol / total_vol) * 100 if total_vol > 0 else 50.0

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_gauge_inner = inset_axes(
        ax_gauge,
        width="90%",
        height="90%",
        loc="center",
        bbox_to_anchor=(0, -0.06, 1, 1),
        bbox_transform=ax_gauge.transAxes,
    )
    _create_gauge_chart(ax_gauge_inner, vp_ratio)
    ax_gauge.set_xticks([])
    ax_gauge.set_yticks([])
    for spine in ax_gauge.spines.values():
        spine.set_visible(False)

    # Sector Context
    ax_sector = fig.add_subplot(gs_right[1, 0])
    ax_sector.set_xlim(0, 10)
    ax_sector.set_ylim(0, 10)
    ax_sector.axis("off")
    ax_sector.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_sector)

    try:
        from .sector_comparison import get_peer_table_data

        peer_data = get_peer_table_data(
            result.ticker, max_peers=2, include_ticker=False
        )
        sector_name = peer_data.get("sector", "Unknown")
        peers = peer_data.get("peers", [])
    except:
        sector_name, peers = "Unknown", []

    ax_sector.text(
        5,
        8.5,
        f"Sector: {sector_name}",
        fontsize=Typography.LG,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        ha="center",
    )

    y_start = 6.5
    ax_sector.text(
        1.5,
        y_start,
        "Ticker",
        fontsize=Typography.SM,
        color=T.TEXT_SECONDARY,
        fontweight="bold",
        ha="center",
    )
    ax_sector.text(
        5,
        y_start,
        "Price",
        fontsize=Typography.SM,
        color=T.TEXT_SECONDARY,
        fontweight="bold",
        ha="center",
    )
    ax_sector.text(
        8,
        y_start,
        "Change",
        fontsize=Typography.SM,
        color=T.TEXT_SECONDARY,
        fontweight="bold",
        ha="center",
    )
    ax_sector.plot(
        [0.5, 9.5],
        [y_start - 0.7, y_start - 0.7],
        color=T.BORDER,
        linewidth=1,
        alpha=0.5,
    )

    change_color = T.BULLISH if result.change_percent >= 0 else T.BEARISH
    ax_sector.text(
        1.5,
        4.8,
        result.ticker,
        fontsize=Typography.BASE,
        color=T.TEXT_WHITE,
        fontweight="bold",
        ha="center",
    )
    ax_sector.text(
        5,
        4.8,
        f"{result.current_price:,.0f}",
        fontsize=Typography.BASE,
        color=T.TEXT_PRIMARY,
        ha="center",
    )
    ax_sector.text(
        8,
        4.8,
        f"{result.change_percent:+.1f}%",
        fontsize=Typography.BASE,
        color=change_color,
        fontweight="bold",
        ha="center",
    )

    y_pos = 2.8
    for peer in peers[:2]:
        peer_color = T.BULLISH if peer.is_up else T.BEARISH
        ax_sector.text(
            1.5,
            y_pos,
            peer.ticker,
            fontsize=Typography.SM,
            color=T.TEXT_SECONDARY,
            ha="center",
        )
        ax_sector.text(
            5,
            y_pos,
            f"{peer.price:,.0f}",
            fontsize=Typography.SM,
            color=T.TEXT_SECONDARY,
            ha="center",
        )
        ax_sector.text(
            8,
            y_pos,
            f"{peer.change_percent:+.1f}%",
            fontsize=Typography.SM,
            color=peer_color,
            ha="center",
        )
        y_pos -= 1.4

    # Key Insights
    ax_insights = fig.add_subplot(gs_right[2, 0])
    ax_insights.set_xlim(0, 10)
    ax_insights.set_ylim(0, 10)
    ax_insights.axis("off")
    ax_insights.set_facecolor(T.BG_SECONDARY)
    _add_card_container(ax_insights)

    ax_insights.text(
        5,
        8.6,
        "Key Insights",
        fontsize=Typography.LG,
        fontweight="bold",
        color=T.TEXT_PRIMARY,
        ha="center",
    )

    insights = []
    if "Bull" in result.trend:
        insights.append(("▲", T.BULLISH, "Bullish trend confirmed"))
    elif "Bear" in result.trend:
        insights.append(("▼", T.BEARISH, "Bearish trend active"))
    else:
        insights.append(("◆", T.NEUTRAL, "Sideways consolidation"))

    if result.rsi > 70:
        insights.append(("⚠", T.BEARISH, f"RSI Overbought ({result.rsi:.1f})"))
    elif result.rsi < 30:
        insights.append(("💎", T.BULLISH, f"RSI Oversold ({result.rsi:.1f})"))
    else:
        insights.append(("◎", T.NEUTRAL, f"RSI Neutral ({result.rsi:.1f})"))

    avg_vol = hist["Volume"].mean()
    last_vol = hist["Volume"].iloc[-1]
    vol_icon = "◉" if last_vol > avg_vol * 1.5 else "○"
    vol_text = "High volume" if last_vol > avg_vol * 1.5 else "Average volume"
    insights.append((vol_icon, T.TEXT_SECONDARY, vol_text))

    if result.current_price > sma_50.iloc[-1]:
        insights.append(("▲", T.BULLISH, "Above SMA 50"))
    else:
        insights.append(("▼", T.BEARISH, "Below SMA 50"))

    y_pos = 6.6
    for icon, color, text in insights:
        ax_insights.text(1.2, y_pos, icon, fontsize=12, color=color, va="center")
        ax_insights.text(
            2.2,
            y_pos,
            text,
            fontsize=Typography.BASE,
            color=T.TEXT_PRIMARY,
            va="center",
        )
        y_pos -= 1.5

    # Save
    if output_path is None:
        output_path = f"{result.ticker}_executive.png"

    plt.savefig(
        output_path,
        dpi=150,
        facecolor=T.BG_PRIMARY,
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


# ============================================================================
# UNIFIED CHART GENERATION API
# ============================================================================
def generate_chart(
    analyzer: "IDXAnalyzer",
    style: Literal["standard", "executive"] = "standard",
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Generate chart with specified style.

    Args:
        analyzer: IDXAnalyzer instance with fetched data
        style: Chart style - 'standard' or 'executive'
        output_path: Output file path (optional)
        show: Whether to display the chart

    Returns:
        Path to generated chart file
    """
    if style == "executive":
        return generate_executive_dashboard(analyzer, output_path, show)
    else:
        return generate_standard_chart(analyzer, output_path, show)
