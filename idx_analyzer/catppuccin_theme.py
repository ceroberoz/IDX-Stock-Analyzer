"""
Catppuccin Mocha color theme for IDX Analyzer charts.
A soothing pastel theme for the high-spirited.
https://github.com/catppuccin/catppuccin
"""

from dataclasses import dataclass


@dataclass
class CatppuccinMocha:
    """Catppuccin Mocha color palette"""

    # Base colors
    BASE = "#1e1e2e"  # Main background
    MANTLE = "#181825"  # Darker background
    CRUST = "#11111b"  # Darkest background

    # Surface colors
    SURFACE0 = "#313244"
    SURFACE1 = "#45475a"
    SURFACE2 = "#585b70"

    # Overlay colors
    OVERLAY0 = "#6c7086"
    OVERLAY1 = "#7f849c"
    OVERLAY2 = "#9399b2"

    # Text colors
    SUBTEXT0 = "#a6adc8"
    SUBTEXT1 = "#bac2de"
    TEXT = "#cdd6f4"

    # Accent colors
    LAVENDER = "#b4befe"
    BLUE = "#89b4fa"
    SAPPHIRE = "#74c7ec"
    SKY = "#89dceb"
    TEAL = "#94e2d5"
    GREEN = "#a6e3a1"
    YELLOW = "#f9e2af"
    PEACH = "#fab387"
    MAROON = "#eba0ac"
    RED = "#f38ba8"
    MAUVE = "#cba6f7"
    PINK = "#f5c2e7"
    FLAMINGO = "#f2cdcd"
    ROSEWATER = "#f5e0dc"


# Chart-specific color mappings
CHART_THEME = {
    # Background
    "bg_facecolor": CatppuccinMocha.BASE,
    "axes_facecolor": CatppuccinMocha.MANTLE,
    "grid_color": CatppuccinMocha.SURFACE0,
    # Surface colors (for spines/borders)
    "surface0": CatppuccinMocha.SURFACE0,
    "surface1": CatppuccinMocha.SURFACE1,
    "surface2": CatppuccinMocha.SURFACE2,
    # Text
    "text": CatppuccinMocha.TEXT,
    "title_color": CatppuccinMocha.TEXT,
    "label_color": CatppuccinMocha.SUBTEXT1,
    "tick_color": CatppuccinMocha.SUBTEXT0,
    "subtext0": CatppuccinMocha.SUBTEXT0,
    "subtext1": CatppuccinMocha.SUBTEXT1,
    # Overlay
    "overlay0": CatppuccinMocha.OVERLAY0,
    "overlay1": CatppuccinMocha.OVERLAY1,
    # Price elements
    "price_line": CatppuccinMocha.BLUE,
    "current_price_line": CatppuccinMocha.TEXT,
    # Moving averages
    "sma_20": CatppuccinMocha.PEACH,
    "sma_50": CatppuccinMocha.MAUVE,
    "sma_200": CatppuccinMocha.RED,
    # Bollinger Bands
    "bb_upper": CatppuccinMocha.OVERLAY0,
    "bb_lower": CatppuccinMocha.OVERLAY0,
    "bb_fill": CatppuccinMocha.SURFACE0,
    # Support/Resistance
    "support": CatppuccinMocha.GREEN,
    "resistance": CatppuccinMocha.RED,
    "support_zone": CatppuccinMocha.GREEN,
    "resistance_zone": CatppuccinMocha.RED,
    # Volume
    "volume": CatppuccinMocha.SAPPHIRE,
    "volume_profile": CatppuccinMocha.LAVENDER,
    "volume_poc": CatppuccinMocha.RED,
    "value_area": CatppuccinMocha.BLUE,
    # RSI
    "rsi_line": CatppuccinMocha.MAUVE,
    "rsi_overbought": CatppuccinMocha.RED,
    "rsi_oversold": CatppuccinMocha.GREEN,
    "rsi_overbought_zone": CatppuccinMocha.RED,
    "rsi_oversold_zone": CatppuccinMocha.GREEN,
    # Peers
    "peer_up": CatppuccinMocha.GREEN,
    "peer_down": CatppuccinMocha.RED,
    "peer_neutral": CatppuccinMocha.YELLOW,
    # Insight panel
    "insight_bg": CatppuccinMocha.MANTLE,
    "insight_border_bull": CatppuccinMocha.GREEN,
    "insight_border_bear": CatppuccinMocha.RED,
    "insight_border_neutral": CatppuccinMocha.YELLOW,
    # Section headers
    "section_header": CatppuccinMocha.LAVENDER,
    "section_icon": CatppuccinMocha.PEACH,
    # Accent colors for general use
    "green": CatppuccinMocha.GREEN,
    "red": CatppuccinMocha.RED,
    "yellow": CatppuccinMocha.YELLOW,
    "blue": CatppuccinMocha.BLUE,
    "mauve": CatppuccinMocha.MAUVE,
    "peach": CatppuccinMocha.PEACH,
    "pink": CatppuccinMocha.PINK,
    "lavender": CatppuccinMocha.LAVENDER,
}


# Icon mappings for sections
ICONS = {
    "price": "◆",  # Diamond for main price
    "trend": "◈",  # Trend direction
    "volume": "◉",  # Volume circle
    "rsi": "◎",  # RSI gauge
    "support": "▼",  # Support (down arrow)
    "resistance": "▲",  # Resistance (up arrow)
    "peers": "◐",  # Peers/half circle
    "sector": "◑",  # Sector
    "signal": "◆",  # Trading signal
    "alert": "◈",  # Alert
    "bull": "▲",  # Bullish
    "bear": "▼",  # Bearish
    "neutral": "◆",  # Neutral
    "up": "▲",  # Up arrow
    "down": "▼",  # Down arrow
    "chart": "◈",  # Chart icon
    "info": "◉",  # Info
    "calendar": "◎",  # Calendar
}


def apply_catppuccin_style(fig, axes):
    """Apply Catppuccin Mocha styling to matplotlib figure and axes."""
    theme = CHART_THEME

    # Figure background
    fig.patch.set_facecolor(theme["bg_facecolor"])

    # Apply to all axes
    for ax in axes:
        ax.set_facecolor(theme["axes_facecolor"])
        ax.tick_params(colors=theme["tick_color"])
        ax.xaxis.label.set_color(theme["label_color"])
        ax.yaxis.label.set_color(theme["label_color"])
        ax.title.set_color(theme["title_color"])

        # Grid
        ax.grid(True, alpha=0.2, color=theme["grid_color"])

        # Spines
        for spine in ax.spines.values():
            spine.set_color(theme["surface0"])

    return fig, axes


def get_trend_color(trend: str) -> str:
    """Get color based on trend description."""
    if "Bull" in trend or "bull" in trend.lower():
        return CHART_THEME["insight_border_bull"]
    elif "Bear" in trend or "bear" in trend.lower():
        return CHART_THEME["insight_border_bear"]
    else:
        return CHART_THEME["insight_border_neutral"]


def get_peer_color(is_up: bool) -> str:
    """Get color for peer based on direction."""
    return CHART_THEME["peer_up"] if is_up else CHART_THEME["peer_down"]
